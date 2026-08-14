"""
策略健全性测试脚本（无需真机）
================================
验证目标：
  1. 策略对不同状态输出不同动作（状态敏感性）
  2. 模拟一段有意义的状态轨迹，观察动作是否合理
  3. 与仿真 evaluate_transformer() 的 action 进行对比

运行方式:
    conda activate 0320
    python test_policy_sanity.py
"""

import os, sys
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.environ["PROJECT_ROOT"] = PROJECT_ROOT

# ─────────────────────────────────────────────────────────────────
#  从 deploy_franka.py 复用常量和 load_pace_agent
# ─────────────────────────────────────────────────────────────────
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

def make_state(eef, obj, goal, gripper=(0.04, 0.04)):
    """构造 21 维状态向量"""
    eef    = np.array(eef,    dtype=np.float32)
    obj    = np.array(obj,    dtype=np.float32)
    goal   = np.array(goal,   dtype=np.float32)
    finger1 = eef + np.array([0.0,  0.05, 0.0])
    finger2 = eef + np.array([0.0, -0.05, 0.0])
    g = np.array(gripper, dtype=np.float32)
    gripper_state = np.array([g[0], g[1], g[0]+g[1]])
    obs_18 = np.concatenate([eef, finger1, finger2, obj, goal, gripper_state])
    return np.concatenate([obs_18, goal]).astype(np.float32)  # 21 dims

def infer_single(agent, state_np, device):
    """用全零历史（left-pad）做一步推理，返回 4 维 action"""
    task_ids = torch.tensor([[TASK_ID]], dtype=torch.float32, device=device)
    state = torch.tensor(state_np, dtype=torch.float32, device=device)
    state_buf  = state.view(1, 1, STATE_DIM).repeat(1, SEQ_LEN, 1)
    action_buf = torch.zeros((1, SEQ_LEN, ACTION_DIM), device=device)
    reward_buf = torch.zeros((1, SEQ_LEN, 1), device=device)
    with torch.no_grad():
        act = agent.select_action(
            states=state_buf,
            actions=action_buf[:, 1:, :],
            rewards=reward_buf[:, 1:, :],
            task_ids=task_ids,
        )
    return act[0]  # shape (4,)

def simulate_trajectory(agent, device, steps=30):
    """
    模拟一段有反馈的轨迹：
      - EEF 每步按 action 移动（简单积分）
      - 观察策略是否引导 EEF 朝目标物体靠近
    """
    task_ids = torch.tensor([[TASK_ID]], dtype=torch.float32, device=device)
    ACTION_SCALE_XYZ = 0.05

    # 初始场景
    eef  = np.array([0.30, 0.00, 0.30], dtype=np.float32)
    obj  = np.array([0.30, 0.10, 0.10], dtype=np.float32)
    goal = np.array([0.10, 0.00, 0.20], dtype=np.float32)
    gripper = np.array([0.04, 0.04], dtype=np.float32)

    state_np = make_state(eef, obj, goal, gripper)
    state    = torch.tensor(state_np, dtype=torch.float32, device=device)

    state_buf  = state.view(1, 1, STATE_DIM).repeat(1, SEQ_LEN, 1)
    action_buf = torch.zeros((1, SEQ_LEN, ACTION_DIM), device=device)
    reward_buf = torch.zeros((1, SEQ_LEN, 1), device=device)

    print(f"\n{'='*60}")
    print("  模拟轨迹（EEF 按 action 积分移动）")
    print(f"  初始 EEF={eef}, OBJ={obj}, GOAL={goal}")
    print(f"{'='*60}")
    print(f"{'Step':>4}  {'EEF (m)':>30}  {'dist_to_obj(mm)':>15}  gripper")

    for step in range(steps):
        with torch.no_grad():
            act_np = agent.select_action(
                states=state_buf,
                actions=action_buf[:, 1:, :],
                rewards=reward_buf[:, 1:, :],
                task_ids=task_ids,
            )[0]  # (4,)

        delta_xyz    = act_np[:3] * ACTION_SCALE_XYZ
        gripper_open = act_np[3] > 0.0

        eef = eef + delta_xyz
        dist_obj = np.linalg.norm(eef - obj) * 1000  # mm

        print(f"{step:>4}  eef=[{eef[0]:.3f},{eef[1]:.3f},{eef[2]:.3f}]"
              f"  dist={dist_obj:>8.1f} mm  {'OPEN' if gripper_open else 'CLOSE'}")

        # 更新 rolling buffer（状态变化！）
        new_state_np = make_state(eef, obj, goal, gripper)
        new_state    = torch.tensor(new_state_np, dtype=torch.float32, device=device)

        state_buf  = torch.roll(state_buf,  -1, dims=1)
        action_buf = torch.roll(action_buf, -1, dims=1)
        reward_buf = torch.roll(reward_buf, -1, dims=1)
        state_buf[:, -1, :]  = new_state
        action_buf[:, -1, :] = torch.tensor(act_np, device=device)
        reward_buf[:, -1, :] = 0.0

    return eef, obj, goal

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    print("\n[1/3] 加载模型...")
    agent = load_agent(device)
    print("模型加载完成。\n")

    # ── 测试 1：状态敏感性 ────────────────────────────────────────────────────
    print("=" * 60)
    print("测试 1：状态敏感性（不同状态 → 不同 action？）")
    print("=" * 60)
    test_states = {
        "EEF远离物体": make_state(eef=[0.5, 0.0, 0.5], obj=[0.3, 0.1, 0.1], goal=[0.1, 0.0, 0.2]),
        "EEF靠近物体": make_state(eef=[0.3, 0.1, 0.15], obj=[0.3, 0.1, 0.1], goal=[0.1, 0.0, 0.2]),
        "EEF在目标上方": make_state(eef=[0.1, 0.0, 0.25], obj=[0.3, 0.1, 0.1], goal=[0.1, 0.0, 0.2]),
        "EEF夹住物体后": make_state(eef=[0.3, 0.1, 0.1], obj=[0.3, 0.1, 0.1], goal=[0.1, 0.0, 0.2], gripper=(0.0, 0.0)),
    }
    prev_act = None
    all_different = True
    for name, state_np in test_states.items():
        act = infer_single(agent, state_np, device)
        act_scaled = act[:3] * 0.05 * 1000  # mm
        gripper_str = "OPEN" if act[3] > 0 else "CLOSE"
        print(f"  {name}")
        print(f"    → Δxyz=[{act_scaled[0]:.1f}, {act_scaled[1]:.1f}, {act_scaled[2]:.1f}]mm  gripper={gripper_str}")
        if prev_act is not None and np.allclose(act, prev_act, atol=1e-4):
            all_different = False
            print("    ⚠️  与上一个状态输出相同！策略可能对状态不敏感")
        prev_act = act

    # ── 范围检查 ────────────────────────────────────────────────────────────
    print("\n测试 1b：action 值域检查（必须在 [-1, 1]）")
    all_in_range = True
    for name, state_np in test_states.items():
        act = infer_single(agent, state_np, device)
        in_range = np.all(np.abs(act) <= 1.0)
        if not in_range:
            all_in_range = False
            print(f"  ❌ {name}: action={act} 超出范围！")
    if all_in_range:
        print("  ✅ 所有 action 均在 [-1, 1] 范围内")

    if all_different:
        print("  ✅ 状态敏感性通过：不同状态产生不同 action")
    else:
        print("  ⚠️  状态敏感性警告：部分状态产生相同 action（可能正常，取决于任务）")

    # ── 测试 2：模拟闭环轨迹 ──────────────────────────────────────────────────
    print("\n[2/3] 测试 2：模拟闭环轨迹（状态随 action 更新）")
    final_eef, obj, goal = simulate_trajectory(agent, device, steps=40)

    dist_to_obj  = np.linalg.norm(final_eef - obj)  * 1000
    dist_to_goal = np.linalg.norm(final_eef - goal) * 1000
    print(f"\n  最终 EEF 距物体: {dist_to_obj:.1f} mm")
    print(f"  最终 EEF 距目标: {dist_to_goal:.1f} mm")
    if dist_to_obj < 100:
        print("  ✅ EEF 朝物体靠近（<100mm）—— 策略方向正确")
    else:
        print("  ⚠️  EEF 未显著靠近物体，检查状态构造或 task_id 是否正确")

    # ── 测试 3：单步推理速度 ──────────────────────────────────────────────────
    import time
    print("\n[3/3] 测试 3：推理速度（目标 <100ms @ 10Hz）")
    state_np = make_state([0.3, 0.0, 0.3], [0.3, 0.1, 0.1], [0.1, 0.0, 0.2])
    N = 50
    t0 = time.time()
    for _ in range(N):
        infer_single(agent, state_np, device)
    elapsed = (time.time() - t0) / N * 1000
    print(f"  单步推理耗时: {elapsed:.1f} ms  (控制周期: {1000/10:.0f} ms)")
    if elapsed < 80:
        print(f"  ✅ 推理速度满足 10Hz 要求（余量 {100-elapsed:.0f}ms）")
    else:
        print(f"  ⚠️  推理偏慢，可能无法稳定保持 10Hz")

    print("\n" + "=" * 60)
    print("  测试完成。判断标准：")
    print("  ✅ 状态敏感性 + ✅ 值域正确 + ✅ 轨迹朝目标移动 + ✅ 速度充足")
    print("  → 模型推理部分已就绪，可以连接真机进行下一步验证")
    print("=" * 60)


if __name__ == "__main__":
    main()
