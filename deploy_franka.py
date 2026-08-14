"""
PACE Real Robot Deployment Script for Franka FR3/Panda
========================================================
将训练好的PACE策略部署到真实Franka机器人上。

使用方法:
    conda activate 0320
    cd /path/to/Predictive-Adaptation-for-Collective-Embodiments-Learning
    python deploy_franka.py

依赖:
    - franka_ros2 / fr3_controller_lin.py 必须在真机Ubuntu上运行
    - 本脚本通过 ROS2 topic 与机器人控制器通信
"""

import os
import sys
import time
import threading
import numpy as np
import torch

# ── 路径设置 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.environ["PROJECT_ROOT"] = PROJECT_ROOT

# ── Checkpoint 配置（根据你的训练结果修改）────────────────────────────────────
SEED = 5   # 选择 seed_3 / seed_4 / seed_5
EXPERIMENT = "task_dyn_true"

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "logs/experiment_test/model_dir",
    EXPERIMENT,
    f"model_col_seed_{SEED}",
)
TRANSFORMER_CKPT_STEP = 60001   # 对应 transformer_encoder_60001.pt

REPR_TRANSFORMER_CKPT = os.path.join(
    PROJECT_ROOT,
    f"Transformer_RNN/checkpoints_{EXPERIMENT}_seed_{SEED}",
    "representation_cls_transformer_checkpoint.pth",
)

# ── 策略超参数（与 config.json 一致，不要改）─────────────────────────────────
SEQ_LEN     = 20       # rolling buffer 长度
STATE_DIM   = 21       # obs[:18] + obs[36:] → 21 维
ACTION_DIM  = 4        # [Δx, Δy, Δz, gripper]
ACTION_RANGE = (-1.0, 1.0)
TASK_ID     = 0        # pick-place-v2 的 task_id (单任务时为 0)
CLS_DIM     = 6        # prediction_head_cls.latent_dim

# ── 真机控制参数 ───────────────────────────────────────────────────────────────
CONTROL_HZ       = 10      # 推理频率 (Hz)，与仿真 max_episode_steps=400 / ~40s 对应
MAX_EPISODE_STEPS = 200    # 每轮最多走多少步（真机保守一些）
ACTION_SCALE_XYZ  = 0.05   # 每步最大移动距离 (m)。先从小值开始！
GRIPPER_THRESHOLD = 0.0    # action[3] > threshold → 张开, < threshold → 闭合

# ── ROS2 Topic 名称（与 fr3_controller_lin.py 对应，按需修改）─────────────────
CARTESIAN_DELTA_TOPIC = "/cartesian_delta_cmd"   # 发给控制器的 Δpose 指令
ROBOT_STATE_TOPIC     = "/robot_state"           # 从控制器读取的机器人状态
OBJECT_POSE_TOPIC     = "/object_pose"           # 从感知节点读取的物体位姿


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  1.  加载 PACE 模型                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def load_pace_agent(device: torch.device):
    """
    直接从训练时保存的 config.json 加载配置，实例化 TransformerAgent。
    完全绕开 Hydra compose API（因为项目用的是 hydra 1.0.7，没有 compose）。
    """
    from omegaconf import OmegaConf, DictConfig
    import hydra

    # ── 1. 直接读取训练时保存的完整配置 ──────────────────────────────────────────
    config_path = os.path.join(PROJECT_ROOT, "logs/experiment_test/config.json")
    cfg = OmegaConf.load(config_path)

    # 修正 project_root（在新机器上路径可能不同）
    OmegaConf.update(cfg, "project_root", PROJECT_ROOT, merge=True)
    OmegaConf.update(cfg, "setup.base_path", PROJECT_ROOT, merge=True)
    OmegaConf.update(cfg, "setup.save_dir",
                     os.path.join(PROJECT_ROOT, "logs/experiment_test"), merge=True)

    # ── 2. 修正 transformer_encoder 内部绝对路径 ──────────────────────────────
    repr_ckpt = os.path.join(
        PROJECT_ROOT,
        f"Transformer_RNN/checkpoints_{EXPERIMENT}_seed_{SEED}",
        "representation_cls_transformer_checkpoint.pth",
    )
    OmegaConf.update(cfg,
        "transformer_collective_network.transformer_encoder.representation_transformer.model_path",
        repr_ckpt, merge=True)
    OmegaConf.update(cfg,
        "transformer_collective_network.transformer_encoder.prediction_head_cls.model_path",
        repr_ckpt, merge=True)

    # predictive_adapter pretrained_dir
    OmegaConf.update(cfg,
        "transformer_collective_network.predictive_adapter.pretrained_dir",
        os.path.join(PROJECT_ROOT, "logs/experiment_test/model_dir"), merge=True)

    tcn_cfg = cfg.transformer_collective_network

    # ── 3. 用 hydra.utils.instantiate 实例化 TransformerAgent ─────────────────
    agent = hydra.utils.instantiate(
        tcn_cfg.builder,
        env_obs_shape=[STATE_DIM],
        action_shape=[ACTION_DIM],
        action_range=ACTION_RANGE,
        device=device,
        actor_cfg=tcn_cfg.actor,
        critic_cfg=tcn_cfg.critic,
        transformer_encoder_cfg=tcn_cfg.transformer_encoder,
        actor_optimizer_cfg=tcn_cfg.optimizers.actor,
        critic_optimizer_cfg=tcn_cfg.optimizers.critic,
        alpha_optimizer_cfg=tcn_cfg.optimizers.alpha,
        transformer_encoder_optimizer_cfg=tcn_cfg.optimizers.transformer_encoder,
        experiment=EXPERIMENT,
        seed=SEED,
    )

    # ── 4. 加载 actor / transformer_encoder checkpoint ────────────────────────
    agent.load(MODEL_DIR, step=TRANSFORMER_CKPT_STEP)
    agent.actor.eval()
    agent.task_encoder.eval()
    if agent.predictive_adapter is not None:
        agent.predictive_adapter.eval()

    print(f"[PACE] Agent loaded from {MODEL_DIR} (step={TRANSFORMER_CKPT_STEP})")
    return agent


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  2.  机器人状态提供器（State Provider）                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class FrankaStateProvider:
    """
    从 ROS2 订阅机器人末端位姿 + 物体位姿，
    组装成与仿真一致的 21 维 state 向量。

    仿真中 obs[:18] + obs[36:] 含义（Meta-World pick-place-v2）：
        obs[0:3]   末端执行器 xyz
        obs[3:6]   手指 1 xyz（近似 = EEF xyz，真机可复用）
        obs[6:9]   手指 2 xyz（近似 = EEF xyz）
        obs[9:12]  目标物体 xyz
        obs[12:15] 另一个对象 / 目标位置 xyz
        obs[15:18] 夹爪开合状态 (3维，实际含义：[qpos1, qpos2, extra])
        obs[36:39] 目标 goal xyz (3维)
    共 21 维。

    ⚠️  这是最关键的桥接部分！需要根据你的手眼标定结果
        将相机坐标系下的物体位置转换到机器人基座坐标系。
    """

    def __init__(self):
        self.eef_pos    = np.zeros(3)   # 末端执行器位置 (m), 机器人基座坐标系
        self.gripper_qpos = np.zeros(2) # 夹爪关节角 (rad)
        self.object_pos = np.zeros(3)   # 目标物体位置, 机器人基座坐标系
        self.goal_pos   = np.zeros(3)   # 目标放置位置, 机器人基座坐标系
        self._lock = threading.Lock()

        # ── 在真机上：在这里初始化 ROS2 subscriber ────────────────────────────
        # import rclpy
        # from geometry_msgs.msg import PoseStamped
        # self.node = rclpy.create_node('pace_state_provider')
        # self.node.create_subscription(PoseStamped, ROBOT_STATE_TOPIC,
        #                               self._robot_state_callback, 10)
        # self.node.create_subscription(PoseStamped, OBJECT_POSE_TOPIC,
        #                               self._object_pose_callback, 10)

    def _robot_state_callback(self, msg):
        """ROS2 回调: 更新末端执行器位置"""
        with self._lock:
            self.eef_pos[0] = msg.pose.position.x
            self.eef_pos[1] = msg.pose.position.y
            self.eef_pos[2] = msg.pose.position.z
            # gripper_qpos 需要从 franka_ros2 的 JointState topic 读取
            # self.gripper_qpos = [joint8_pos, joint9_pos]

    def _object_pose_callback(self, msg):
        """ROS2 回调: 更新物体位置（来自感知节点）"""
        with self._lock:
            self.object_pos[0] = msg.pose.position.x
            self.object_pos[1] = msg.pose.position.y
            self.object_pos[2] = msg.pose.position.z

    def set_goal(self, goal_pos: np.ndarray):
        """在 episode 开始时设置目标位置（可以手动输入或从任务定义获取）"""
        with self._lock:
            self.goal_pos = goal_pos.copy()

    def get_state_21d(self) -> np.ndarray:
        """
        返回与仿真一致的 21 维 state 向量。

        仿真 obs 压缩逻辑（见 collective_metaworld.py line 222）：
            state = np.concatenate([obs[:18], obs[36:39]])  → 21 维

        真机对应的构造方式:
            [eef_xyz(3), finger1_xyz(3), finger2_xyz(3),
             object_xyz(3), goal_xyz(3), gripper_state(3),
             goal_xyz_dup(3)]    → 21 维
        """
        with self._lock:
            eef   = self.eef_pos.copy()
            obj   = self.object_pos.copy()
            goal  = self.goal_pos.copy()
            g_qpos = self.gripper_qpos.copy()

        # finger1 ≈ finger2 ≈ eef（真机夹爪偏移较小，可以用 eef 近似）
        finger1 = eef + np.array([0.0, 0.05, 0.0])   # 根据实际夹爪几何调整
        finger2 = eef + np.array([0.0, -0.05, 0.0])

        # gripper_state: [qpos1, qpos2, normalized_opening]
        gripper_state = np.array([
            g_qpos[0],
            g_qpos[1],
            (g_qpos[0] + g_qpos[1]),   # 开合程度
        ])

        # 对应 obs[:18]
        obs_18 = np.concatenate([
            eef,           # [0:3]
            finger1,       # [3:6]
            finger2,       # [6:9]
            obj,           # [9:12]
            goal,          # [12:15]
            gripper_state, # [15:18]
        ])

        # 对应 obs[36:39]（goal 位置的另一份拷贝）
        obs_3 = goal  # [36:39]

        state_21d = np.concatenate([obs_18, obs_3]).astype(np.float32)
        assert state_21d.shape == (21,), f"State dim error: {state_21d.shape}"
        return state_21d

    # ── 调试用：直接手动设定状态（不依赖ROS2，测试策略逻辑时使用）─────────────
    def set_mock_state(self, eef, obj, goal, gripper=None):
        with self._lock:
            self.eef_pos    = np.array(eef, dtype=np.float32)
            self.object_pos = np.array(obj, dtype=np.float32)
            self.goal_pos   = np.array(goal, dtype=np.float32)
            if gripper is not None:
                self.gripper_qpos = np.array(gripper, dtype=np.float32)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  3.  机器人执行器（Action Executor）                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class FrankaActionExecutor:
    """
    接收 PACE 输出的 4 维 action [Δx, Δy, Δz, gripper]，
    发送给 fr3_controller_lin.py 控制器。

    action 范围: [-1, 1]
    真机执行: Δxyz 需要乘以 ACTION_SCALE_XYZ 换算成实际位移 (m)
    """

    def __init__(self):
        # ── 在真机上：在这里初始化 ROS2 publisher ────────────────────────────
        # import rclpy
        # from geometry_msgs.msg import Twist
        # self.node = rclpy.create_node('pace_action_executor')
        # self.pub = self.node.create_publisher(Twist, CARTESIAN_DELTA_TOPIC, 10)
        pass

    def execute(self, action: np.ndarray):
        """
        执行一步 action。

        Args:
            action: shape (4,), 范围 [-1, 1]
                    [Δx, Δy, Δz, gripper_cmd]
        """
        assert action.shape == (4,), f"Action shape error: {action.shape}"

        delta_xyz = action[:3] * ACTION_SCALE_XYZ  # 换算成米
        gripper_open = action[3] > GRIPPER_THRESHOLD

        # ── 真机：发送 ROS2 消息 ──────────────────────────────────────────────
        # from geometry_msgs.msg import Twist
        # msg = Twist()
        # msg.linear.x = float(delta_xyz[0])
        # msg.linear.y = float(delta_xyz[1])
        # msg.linear.z = float(delta_xyz[2])
        # msg.angular.z = 1.0 if gripper_open else -1.0  # 夹爪控制约定
        # self.pub.publish(msg)

        # ── 调试模式：只打印 ───────────────────────────────────────────────────
        gripper_str = "OPEN" if gripper_open else "CLOSE"
        print(f"  ACTION → Δxyz={[round(v*1000,1) for v in delta_xyz]} mm | gripper={gripper_str}")

    def open_gripper(self):
        print("  GRIPPER → OPEN")
        # 发送张开指令

    def close_gripper(self):
        print("  GRIPPER → CLOSE")
        # 发送闭合指令

    def move_to_home(self):
        """回到 home 姿态（每个 episode 开始前调用）"""
        print("[Robot] Moving to home position...")
        # 发送 home 指令或调用 franka_ros2 的 move_to_start 服务
        time.sleep(2.0)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  4.  主推理循环（Rolling Buffer Inference）                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class PACEDeployer:
    """
    完整的 PACE 真机部署器，实现与 evaluate_transformer() 相同的 rolling buffer 推理。
    """

    def __init__(self, agent, state_provider: FrankaStateProvider,
                 action_executor: FrankaActionExecutor, device: torch.device):
        self.agent    = agent
        self.state_prov = state_provider
        self.executor   = action_executor
        self.device     = device

        # task_id tensor: shape [1, 1]（单任务，task_id=0）
        self.task_ids = torch.tensor([[TASK_ID]], dtype=torch.float32, device=device)

    def run_episode(self, episode_idx: int = 0):
        """运行一个完整的 episode（与 evaluate_transformer() 逻辑完全一致）"""

        print(f"\n{'='*60}")
        print(f" Episode {episode_idx}  |  max_steps={MAX_EPISODE_STEPS}")
        print(f"{'='*60}")

        # 1. 回 home，等待稳定
        self.executor.move_to_home()
        time.sleep(0.5)

        # 2. 获取初始 state，初始化 rolling buffer（与仿真的 left_pad 一致）
        first_state_np = self.state_prov.get_state_21d()
        first_state = torch.tensor(first_state_np, dtype=torch.float32,
                                   device=self.device).unsqueeze(0)  # [1, 21]

        # state_buffer:  [1, SEQ_LEN, 21] — 全部填充为初始帧
        state_buffer  = first_state.unsqueeze(1).repeat(1, SEQ_LEN, 1)
        # action_buffer: [1, SEQ_LEN, 4]  — 全零
        action_buffer = torch.zeros((1, SEQ_LEN, ACTION_DIM), device=self.device)
        # reward_buffer: [1, SEQ_LEN, 1]  — 全零（真机无reward，填0即可）
        reward_buffer = torch.zeros((1, SEQ_LEN, 1), device=self.device)

        control_period = 1.0 / CONTROL_HZ

        for step in range(MAX_EPISODE_STEPS):
            t_start = time.time()

            # ── 3. 推理：取 buffer 的特定 slice（与仿真一致）──────────────────
            with torch.no_grad():
                active_states  = state_buffer            # [1, SEQ_LEN, 21]
                active_actions = action_buffer[:, 1:, :] # [1, SEQ_LEN-1, 4]
                active_rewards = reward_buffer[:, 1:, :] # [1, SEQ_LEN-1, 1]

                action_out = self.agent.select_action(   # 确定性动作（不采样）
                    states=active_states,
                    actions=active_actions,
                    rewards=active_rewards,
                    task_ids=self.task_ids,
                )
            # action_out: numpy [1, 4] → [4]
            action_np = action_out[0]

            print(f"Step {step:3d}:", end="")
            # ── 4. 执行动作 ───────────────────────────────────────────────────
            self.executor.execute(action_np)

            # ── 5. 读取新状态 ──────────────────────────────────────────────────
            time.sleep(max(0, control_period - (time.time() - t_start)))
            new_state_np = self.state_prov.get_state_21d()
            new_state = torch.tensor(new_state_np, dtype=torch.float32,
                                     device=self.device).unsqueeze(0)  # [1, 21]

            # reward 真机填 0（不影响推理，rolling buffer 需要维持形状）
            new_reward = torch.zeros((1, 1), device=self.device)

            # ── 6. Rolling buffer 更新（与仿真完全一致）────────────────────────
            state_buffer  = torch.roll(state_buffer,  shifts=-1, dims=1)
            action_buffer = torch.roll(action_buffer, shifts=-1, dims=1)
            reward_buffer = torch.roll(reward_buffer, shifts=-1, dims=1)

            state_buffer[:, -1, :]  = new_state
            action_buffer[:, -1, :] = torch.tensor(
                action_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            reward_buffer[:, -1, :] = new_reward

            # ── 7. 成功判断（根据任务自行添加）──────────────────────────────────
            # 例：pick-place-v2 成功条件 = 物体与目标距离 < 0.05m
            obj_pos  = self.state_prov.object_pos
            goal_pos = self.state_prov.goal_pos
            dist = np.linalg.norm(obj_pos - goal_pos)
            if dist < 0.05:
                print(f"\n✅ SUCCESS at step {step}! dist={dist:.3f}m")
                return True

        print(f"\n❌ Episode ended after {MAX_EPISODE_STEPS} steps (no success).")
        return False


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  5.  入口                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    print("=" * 60)
    print("  PACE Real Robot Deployment")
    print(f"  Seed: {SEED}  |  Experiment: {EXPERIMENT}")
    print(f"  Model dir: {MODEL_DIR}")
    print("=" * 60)

    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ── 初始化 ROS2（真机上取消注释）────────────────────────────────────────────
    # import rclpy
    # rclpy.init()

    # ── 加载 PACE 模型 ─────────────────────────────────────────────────────────
    print("\n[1/3] Loading PACE agent...")
    agent = load_pace_agent(device)

    # ── 初始化 State Provider 和 Action Executor ──────────────────────────────
    print("[2/3] Initializing robot interface...")
    state_provider  = FrankaStateProvider()
    action_executor = FrankaActionExecutor()

    # ── 设置任务目标（pick-place-v2 的放置目标位置）──────────────────────────────
    # ⚠️  根据你实际的任务场景设置 goal 位置（机器人基座坐标系，单位 m）
    GOAL_POSITION = np.array([0.1, 0.0, 0.2])   # TODO: 修改为实际目标位置
    state_provider.set_goal(GOAL_POSITION)

    # ── (调试模式) 设置初始 mock 状态 ──────────────────────────────────────────
    # 真机上请改为真实 ROS2 订阅，删除下面这行
    state_provider.set_mock_state(
        eef    = [0.3, 0.0, 0.3],   # 末端执行器初始位置
        obj    = [0.3, 0.1, 0.1],   # 物体位置
        goal   = GOAL_POSITION,
        gripper= [0.04, 0.04],      # 夹爪打开状态
    )

    # ── 创建 Deployer ──────────────────────────────────────────────────────────
    print("[3/3] Starting deployment loop...\n")
    deployer = PACEDeployer(agent, state_provider, action_executor, device)

    # ── 运行多个 episode ────────────────────────────────────────────────────────
    NUM_EPISODES = 5
    success_count = 0
    for ep in range(NUM_EPISODES):
        success = deployer.run_episode(episode_idx=ep)
        if success:
            success_count += 1
        time.sleep(2.0)  # episode 间隔

    print(f"\n{'='*60}")
    print(f"  Final Result: {success_count}/{NUM_EPISODES} episodes succeeded")
    print(f"{'='*60}")

    # rclpy.shutdown()


if __name__ == "__main__":
    main()
