"""
ROS2 消息通路测试脚本（在真机 Ubuntu 22.04 上运行）
=====================================================
目的：
  在不启动真实机器人的情况下，验证 ROS2 话题收发逻辑是否正常。
  本脚本会：
    1. 创建 subscriber，监听 /robot_state 和 /object_pose
    2. 创建 publisher，向 /cartesian_delta_cmd 发送 PACE 的 action
    3. 配合 "假数据发布脚本" 完成端到端消息测试

运行方法（真机上，需要 4 个终端）：

  [终端 1] 发布假机器人状态:
    source /opt/ros/humble/setup.bash
    ros2 topic pub /robot_state geometry_msgs/msg/PoseStamped \
      "{header: {stamp: {sec: 0}, frame_id: 'base'}, \
        pose: {position: {x: 0.3, y: 0.0, z: 0.3}, orientation: {w: 1.0}}}" \
      --rate 10

  [终端 2] 发布假物体位置:
    source /opt/ros/humble/setup.bash
    ros2 topic pub /object_pose geometry_msgs/msg/PoseStamped \
      "{header: {stamp: {sec: 0}, frame_id: 'base'}, \
        pose: {position: {x: 0.3, y: 0.1, z: 0.1}, orientation: {w: 1.0}}}" \
      --rate 10

  [终端 3] 运行本脚本:
    source /opt/ros/humble/setup.bash
    conda activate 0320
    python test_ros2_comms.py

  [终端 4] 监听输出动作:
    source /opt/ros/humble/setup.bash
    ros2 topic echo /cartesian_delta_cmd

预期结果：
  - 终端 3 中看到 "[✅ RX robot_state]" 和 "[✅ RX object_pose]" 日志
  - 终端 4 中看到 PACE 输出的 Twist 消息（linear.x/y/z = 动作值）
"""

import os
import sys
import time
import threading
import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.environ["PROJECT_ROOT"] = PROJECT_ROOT

# ── ROS2 初始化 ───────────────────────────────────────────────────────────────
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped, Twist
    from sensor_msgs.msg import JointState
    ROS2_AVAILABLE = True
except ImportError:
    print("❌ rclpy 未找到！请先 source /opt/ros/humble/setup.bash")
    print("   然后重新运行本脚本")
    sys.exit(1)

# ── 常量（与 deploy_franka.py 保持一致）───────────────────────────────────────
SEED              = 5
EXPERIMENT        = "task_dyn_true"
SEQ_LEN           = 20
STATE_DIM         = 21
ACTION_DIM        = 4
ACTION_RANGE      = (-1.0, 1.0)
TASK_ID           = 0
MODEL_DIR = os.path.join(
    PROJECT_ROOT, "logs/experiment_test/model_dir",
    EXPERIMENT, f"model_col_seed_{SEED}",
)
TRANSFORMER_CKPT_STEP = 60001

CARTESIAN_DELTA_TOPIC = "/cartesian_delta_cmd"
ROBOT_STATE_TOPIC     = "/robot_state"
OBJECT_POSE_TOPIC     = "/object_pose"
GRIPPER_JOINT_TOPIC   = "/joint_states"   # 如果需要夹爪状态

ACTION_SCALE_XYZ  = 0.05
GRIPPER_THRESHOLD = 0.0
CONTROL_HZ        = 10
MAX_STEPS         = 30   # 本测试只跑 30 步，验证消息通路即可


# ════════════════════════════════════════════════════════════════════════════════
# ROS2 通信节点
# ════════════════════════════════════════════════════════════════════════════════

class PACECommsNode(Node):
    """
    单一 ROS2 节点，同时负责：
      - 订阅机器人末端位姿（/robot_state）
      - 订阅物体位姿（/object_pose）
      - 发布动作指令（/cartesian_delta_cmd）
    """

    def __init__(self):
        super().__init__('pace_comms_test')
        self._lock = threading.Lock()

        # 内部状态（从 ROS2 消息更新）
        self.eef_pos      = np.array([0.3, 0.0, 0.3], dtype=np.float32)   # 默认初始值
        self.object_pos   = np.array([0.3, 0.1, 0.1], dtype=np.float32)
        self.goal_pos     = np.array([0.1, 0.0, 0.2], dtype=np.float32)   # 由外部设置
        self.gripper_qpos = np.array([0.04, 0.04], dtype=np.float32)

        # 收到消息的计数（用于验证）
        self.robot_state_count  = 0
        self.object_pose_count  = 0

        # ── Subscriber ────────────────────────────────────────────────────────
        self.robot_state_sub = self.create_subscription(
            PoseStamped,
            ROBOT_STATE_TOPIC,
            self._robot_state_callback,
            10,
        )
        self.object_pose_sub = self.create_subscription(
            PoseStamped,
            OBJECT_POSE_TOPIC,
            self._object_pose_callback,
            10,
        )
        # 可选：订阅 JointState 获取夹爪角度
        # self.joint_state_sub = self.create_subscription(
        #     JointState,
        #     GRIPPER_JOINT_TOPIC,
        #     self._joint_state_callback,
        #     10,
        # )

        # ── Publisher ─────────────────────────────────────────────────────────
        self.action_pub = self.create_publisher(
            Twist,
            CARTESIAN_DELTA_TOPIC,
            10,
        )

        self.get_logger().info("✅ PACECommsNode 初始化完成")
        self.get_logger().info(f"   监听: {ROBOT_STATE_TOPIC}, {OBJECT_POSE_TOPIC}")
        self.get_logger().info(f"   发布: {CARTESIAN_DELTA_TOPIC}")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _robot_state_callback(self, msg: PoseStamped):
        with self._lock:
            self.eef_pos[0] = msg.pose.position.x
            self.eef_pos[1] = msg.pose.position.y
            self.eef_pos[2] = msg.pose.position.z
            self.robot_state_count += 1
        if self.robot_state_count <= 3 or self.robot_state_count % 20 == 0:
            self.get_logger().info(
                f"[✅ RX robot_state #{self.robot_state_count}] "
                f"EEF=({self.eef_pos[0]:.3f}, {self.eef_pos[1]:.3f}, {self.eef_pos[2]:.3f})"
            )

    def _joint_state_callback(self, msg: JointState):
        """
        从 JointState 读取夹爪关节角度。
        Franka 夹爪关节名称通常为 'panda_finger_joint1' / 'panda_finger_joint2'
        """
        with self._lock:
            try:
                idx1 = msg.name.index('panda_finger_joint1')
                idx2 = msg.name.index('panda_finger_joint2')
                self.gripper_qpos[0] = msg.position[idx1]
                self.gripper_qpos[1] = msg.position[idx2]
            except ValueError:
                pass  # 关节名不在列表中，跳过

    def _object_pose_callback(self, msg: PoseStamped):
        with self._lock:
            self.object_pos[0] = msg.pose.position.x
            self.object_pos[1] = msg.pose.position.y
            self.object_pos[2] = msg.pose.position.z
            self.object_pose_count += 1
        if self.object_pose_count <= 3 or self.object_pose_count % 20 == 0:
            self.get_logger().info(
                f"[✅ RX object_pose #{self.object_pose_count}] "
                f"OBJ=({self.object_pos[0]:.3f}, {self.object_pos[1]:.3f}, {self.object_pos[2]:.3f})"
            )

    # ── Getters ───────────────────────────────────────────────────────────────

    def get_state_21d(self) -> np.ndarray:
        """返回 21 维状态向量（与 deploy_franka.py 完全一致）"""
        with self._lock:
            eef    = self.eef_pos.copy()
            obj    = self.object_pos.copy()
            goal   = self.goal_pos.copy()
            g_qpos = self.gripper_qpos.copy()

        finger1 = eef + np.array([0.0,  0.05, 0.0])
        finger2 = eef + np.array([0.0, -0.05, 0.0])
        gripper_state = np.array([g_qpos[0], g_qpos[1], g_qpos[0] + g_qpos[1]])

        obs_18 = np.concatenate([eef, finger1, finger2, obj, goal, gripper_state])
        state_21d = np.concatenate([obs_18, goal]).astype(np.float32)
        return state_21d

    # ── Publisher ─────────────────────────────────────────────────────────────

    def publish_action(self, action: np.ndarray):
        """将 PACE 4 维 action 发布为 ROS2 Twist 消息"""
        delta_xyz    = action[:3] * ACTION_SCALE_XYZ
        gripper_open = action[3] > GRIPPER_THRESHOLD

        msg = Twist()
        msg.linear.x = float(delta_xyz[0])
        msg.linear.y = float(delta_xyz[1])
        msg.linear.z = float(delta_xyz[2])
        msg.angular.z = 1.0 if gripper_open else -1.0   # 约定：1=OPEN, -1=CLOSE

        self.action_pub.publish(msg)
        return delta_xyz, gripper_open


# ════════════════════════════════════════════════════════════════════════════════
# 加载 PACE 模型
# ════════════════════════════════════════════════════════════════════════════════

def load_agent(device):
    from omegaconf import OmegaConf
    import hydra
    config_path = os.path.join(PROJECT_ROOT, "logs/experiment_test/config.json")
    cfg = OmegaConf.load(config_path)
    OmegaConf.update(cfg, "project_root", PROJECT_ROOT, merge=True)
    OmegaConf.update(cfg, "setup.base_path", PROJECT_ROOT, merge=True)
    OmegaConf.update(cfg, "setup.save_dir",
                     os.path.join(PROJECT_ROOT, "logs/experiment_test"), merge=True)
    repr_ckpt = os.path.join(
        PROJECT_ROOT,
        f"Transformer_RNN/checkpoints_{EXPERIMENT}_seed_{SEED}",
        "representation_cls_transformer_checkpoint.pth",
    )
    OmegaConf.update(cfg, "transformer_collective_network.transformer_encoder.representation_transformer.model_path", repr_ckpt, merge=True)
    OmegaConf.update(cfg, "transformer_collective_network.transformer_encoder.prediction_head_cls.model_path", repr_ckpt, merge=True)
    OmegaConf.update(cfg, "transformer_collective_network.predictive_adapter.pretrained_dir",
                     os.path.join(PROJECT_ROOT, "logs/experiment_test/model_dir"), merge=True)
    tcn_cfg = cfg.transformer_collective_network
    agent = hydra.utils.instantiate(
        tcn_cfg.builder,
        env_obs_shape=[STATE_DIM], action_shape=[ACTION_DIM],
        action_range=ACTION_RANGE, device=device,
        actor_cfg=tcn_cfg.actor, critic_cfg=tcn_cfg.critic,
        transformer_encoder_cfg=tcn_cfg.transformer_encoder,
        actor_optimizer_cfg=tcn_cfg.optimizers.actor,
        critic_optimizer_cfg=tcn_cfg.optimizers.critic,
        alpha_optimizer_cfg=tcn_cfg.optimizers.alpha,
        transformer_encoder_optimizer_cfg=tcn_cfg.optimizers.transformer_encoder,
        experiment=EXPERIMENT, seed=SEED,
    )
    agent.load(MODEL_DIR, step=TRANSFORMER_CKPT_STEP)
    agent.actor.eval()
    agent.task_encoder.eval()
    if agent.predictive_adapter is not None:
        agent.predictive_adapter.eval()
    return agent


# ════════════════════════════════════════════════════════════════════════════════
# 主测试循环
# ════════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  PACE ROS2 消息通路测试")
    print(f"  监听: {ROBOT_STATE_TOPIC}, {OBJECT_POSE_TOPIC}")
    print(f"  发布: {CARTESIAN_DELTA_TOPIC}")
    print(f"  测试步数: {MAX_STEPS} 步 @ {CONTROL_HZ}Hz")
    print("=" * 60)

    # ── 初始化 ROS2 ─────────────────────────────────────────────────────────
    rclpy.init()
    node = PACECommsNode()

    # 设置目标位置
    node.goal_pos = np.array([0.1, 0.0, 0.2], dtype=np.float32)

    # ── 在后台线程中 spin ROS2（接收消息）────────────────────────────────────
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # ── 等待第一条 ROS2 消息（最多等 10 秒）──────────────────────────────────
    print("\n等待 ROS2 消息...")
    print("（如果长时间无消息，请检查终端 1/2 中的 'ros2 topic pub' 是否在运行）\n")
    t_wait = time.time()
    while node.robot_state_count == 0 or node.object_pose_count == 0:
        time.sleep(0.1)
        if time.time() - t_wait > 10:
            print("⚠️  超过 10 秒没有收到 ROS2 消息！")
            print(f"   robot_state 收到: {node.robot_state_count} 条")
            print(f"   object_pose 收到: {node.object_pose_count} 条")
            print("   请确认:")
            print("   1. source /opt/ros/humble/setup.bash 已执行")
            print("   2. 'ros2 topic pub /robot_state ...' 在另一终端运行中")
            print("   3. ROS_DOMAIN_ID 配置一致")
            rclpy.shutdown()
            return

    print(f"✅ 成功接收到 ROS2 消息！")
    print(f"   robot_state: {node.robot_state_count} 条")
    print(f"   object_pose: {node.object_pose_count} 条")

    # ── 加载 PACE 模型 ────────────────────────────────────────────────────────
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n加载 PACE 模型 (device={device})...")
    agent = load_agent(device)
    print("✅ 模型加载完成\n")

    # ── Rolling buffer 初始化 ─────────────────────────────────────────────────
    task_ids = torch.tensor([[TASK_ID]], dtype=torch.float32, device=device)
    first_state = torch.tensor(
        node.get_state_21d(), dtype=torch.float32, device=device
    )
    state_buf  = first_state.unsqueeze(0).unsqueeze(0).repeat(1, SEQ_LEN, 1)
    action_buf = torch.zeros((1, SEQ_LEN, ACTION_DIM), device=device)
    reward_buf = torch.zeros((1, SEQ_LEN, 1), device=device)

    control_period = 1.0 / CONTROL_HZ

    print(f"{'='*60}")
    print(f"  开始控制循环 ({MAX_STEPS} 步)")
    print(f"{'='*60}")
    print(f"{'Step':>4}  {'EEF (m)':>32}  {'Δxyz (mm)':>28}  gripper")

    for step in range(MAX_STEPS):
        t0 = time.time()

        # ── 推理 ──────────────────────────────────────────────────────────────
        with torch.no_grad():
            action_np = agent.select_action(
                states=state_buf,
                actions=action_buf[:, 1:, :],
                rewards=reward_buf[:, 1:, :],
                task_ids=task_ids,
            )[0]   # shape (4,)

        # ── 发布 action 到 ROS2 ────────────────────────────────────────────────
        delta_xyz, gripper_open = node.publish_action(action_np)
        gripper_str = "OPEN" if gripper_open else "CLOSE"

        # 读取当前状态
        eef = node.eef_pos.copy()
        print(
            f"Step {step:3d}  "
            f"eef=[{eef[0]:.3f},{eef[1]:.3f},{eef[2]:.3f}]  "
            f"Δxyz=[{delta_xyz[0]*1000:+6.1f},{delta_xyz[1]*1000:+6.1f},{delta_xyz[2]*1000:+6.1f}]mm  "
            f"{gripper_str}"
        )

        # ── 等待控制周期 ───────────────────────────────────────────────────────
        elapsed = time.time() - t0
        sleep_t = max(0.0, control_period - elapsed)
        time.sleep(sleep_t)

        # ── 读取新状态，更新 rolling buffer ────────────────────────────────────
        new_state = torch.tensor(
            node.get_state_21d(), dtype=torch.float32, device=device
        )
        state_buf  = torch.roll(state_buf,  -1, dims=1)
        action_buf = torch.roll(action_buf, -1, dims=1)
        reward_buf = torch.roll(reward_buf, -1, dims=1)
        state_buf[:, -1, :]  = new_state
        action_buf[:, -1, :] = torch.tensor(action_np, device=device)
        reward_buf[:, -1, :] = 0.0

    print(f"\n{'='*60}")
    print(f"  ✅ 消息通路测试完成！")
    print(f"  总计接收 robot_state: {node.robot_state_count} 条")
    print(f"  总计接收 object_pose: {node.object_pose_count} 条")
    print(f"  总计发布 action:      {MAX_STEPS} 条")
    print(f"\n  接下来:")
    print(f"  1. 在终端 4 中确认 /cartesian_delta_cmd 有数据输出")
    print(f"  2. 检查 linear.x/y/z 的量级是否合理（应在 ±0.05m 以内）")
    print(f"  3. 如果以上正常，可以连接真实 fr3_controller_lin.py 进行真机测试")
    print(f"{'='*60}")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
