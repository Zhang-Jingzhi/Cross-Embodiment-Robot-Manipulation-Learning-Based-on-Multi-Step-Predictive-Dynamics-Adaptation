"""
ROS2 消息通路本地验证脚本（单进程，无需真机，无需多终端）
==========================================================
原理：
  在同一个进程里同时创建一个 Publisher 节点和一个 Subscriber 节点，
  用 MultiThreadedExecutor 驱动它们。Publisher 每 100ms 发一条消息，
  Subscriber 收到后记录日志。这样完全在本机验证 ROS2 收发是否正常。

运行方法（只需一个终端）：
  conda run -n ros2_humble python test_ros2_local.py

如果看到：
  ✅ [SUB] 收到 PoseStamped: EEF=(0.300, 0.000, 0.300)
  ✅ [SUB] 收到 Twist action: Δxyz=(+0.029, -0.019, -0.029)m  gripper=OPEN
  ...
  ✅ 测试通过！共收到 N 条消息

说明 ROS2 pub/sub 通路在本机完全正常，真机上也可以工作。
"""

import time
import threading
import sys

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from geometry_msgs.msg import PoseStamped, Twist
except ImportError as e:
    print(f"❌ ROS2 导入失败: {e}")
    print("   请确认在 ros2_humble 环境下运行：")
    print("   conda run -n ros2_humble python test_ros2_local.py")
    sys.exit(1)

# ── 测试参数 ──────────────────────────────────────────────────────────────────
ROBOT_STATE_TOPIC     = "/robot_state"
OBJECT_POSE_TOPIC     = "/object_pose"
CARTESIAN_DELTA_TOPIC = "/cartesian_delta_cmd"
PUBLISH_HZ            = 10       # 模拟控制频率 10Hz
TEST_DURATION_SEC     = 5.0      # 测试持续 5 秒


# ════════════════════════════════════════════════════════════════════════════════
# Publisher 节点：模拟机器人状态发布者（真机上由 franka_ros2 驱动）
# ════════════════════════════════════════════════════════════════════════════════
class FakeRobotPublisher(Node):
    def __init__(self):
        super().__init__('fake_robot_publisher')
        self.robot_pub  = self.create_publisher(PoseStamped, ROBOT_STATE_TOPIC, 10)
        self.object_pub = self.create_publisher(PoseStamped, OBJECT_POSE_TOPIC, 10)
        period = 1.0 / PUBLISH_HZ
        self.timer = self.create_timer(period, self._publish)
        self._step = 0
        self.get_logger().info(f"[PUB] 开始发布 {ROBOT_STATE_TOPIC} 和 {OBJECT_POSE_TOPIC} @ {PUBLISH_HZ}Hz")

    def _publish(self):
        # 模拟 EEF 位置（每步轻微变化，模拟机器人在运动）
        t = self._step * 0.01
        robot_msg = PoseStamped()
        robot_msg.header.frame_id = 'base'
        robot_msg.pose.position.x = 0.3 + t * 0.01
        robot_msg.pose.position.y = 0.0
        robot_msg.pose.position.z = 0.3
        robot_msg.pose.orientation.w = 1.0
        self.robot_pub.publish(robot_msg)

        # 模拟物体位置（固定）
        obj_msg = PoseStamped()
        obj_msg.header.frame_id = 'base'
        obj_msg.pose.position.x = 0.3
        obj_msg.pose.position.y = 0.1
        obj_msg.pose.position.z = 0.1
        obj_msg.pose.orientation.w = 1.0
        self.object_pub.publish(obj_msg)

        self._step += 1


# ═════════════════════════��══════════════════════════════════════════════════════
# Action Publisher 节点：模拟 PACE 输出动作（真机上由 deploy_franka.py 驱动）
# ════════════════════════════════════════════════════════════════════════════════
class FakeActionPublisher(Node):
    def __init__(self):
        super().__init__('fake_action_publisher')
        self.action_pub = self.create_publisher(Twist, CARTESIAN_DELTA_TOPIC, 10)
        period = 1.0 / PUBLISH_HZ
        self.timer = self.create_timer(period, self._publish)
        self._step = 0
        self.get_logger().info(f"[PUB] 开始发布 {CARTESIAN_DELTA_TOPIC} @ {PUBLISH_HZ}Hz")

    def _publish(self):
        # 模拟 PACE 输出的 4 维 action 转换后的 Twist
        msg = Twist()
        msg.linear.x =  0.029   # Δx (m)，模拟 PACE 推理结果
        msg.linear.y = -0.019   # Δy (m)
        msg.linear.z = -0.029   # Δz (m)
        msg.angular.z = -1.0    # -1=CLOSE, 1=OPEN
        self.action_pub.publish(msg)
        self._step += 1


# ════════════════════════════════════════════════════════════════════════════════
# Subscriber 节点：监听所有话题，统计收到的消息数
# ════════════════════════════════════════════════════════════════════════════════
class CommsVerifier(Node):
    def __init__(self):
        super().__init__('comms_verifier')
        self.robot_count  = 0
        self.object_count = 0
        self.action_count = 0

        self.create_subscription(PoseStamped, ROBOT_STATE_TOPIC,  self._on_robot,  10)
        self.create_subscription(PoseStamped, OBJECT_POSE_TOPIC,  self._on_object, 10)
        self.create_subscription(Twist,       CARTESIAN_DELTA_TOPIC, self._on_action, 10)

    def _on_robot(self, msg):
        self.robot_count += 1
        if self.robot_count <= 3 or self.robot_count % 10 == 0:
            p = msg.pose.position
            self.get_logger().info(
                f"[SUB] 收到 robot_state #{self.robot_count}: "
                f"EEF=({p.x:.3f}, {p.y:.3f}, {p.z:.3f})"
            )

    def _on_object(self, msg):
        self.object_count += 1
        if self.object_count <= 3 or self.object_count % 10 == 0:
            p = msg.pose.position
            self.get_logger().info(
                f"[SUB] 收到 object_pose #{self.object_count}: "
                f"OBJ=({p.x:.3f}, {p.y:.3f}, {p.z:.3f})"
            )

    def _on_action(self, msg):
        self.action_count += 1
        if self.action_count <= 3 or self.action_count % 10 == 0:
            gripper = "OPEN" if msg.angular.z > 0 else "CLOSE"
            self.get_logger().info(
                f"[SUB] 收到 action #{self.action_count}: "
                f"Δxyz=({msg.linear.x:+.3f}, {msg.linear.y:+.3f}, {msg.linear.z:+.3f})m  "
                f"gripper={gripper}"
            )


# ════════════════════════════════════════════════════════════════════════════════
# 主函数
# ════════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  ROS2 消息通路本地验证")
    print(f"  测试时长: {TEST_DURATION_SEC} 秒 @ {PUBLISH_HZ}Hz")
    print("=" * 60)

    rclpy.init()

    pub_robot  = FakeRobotPublisher()
    pub_action = FakeActionPublisher()
    verifier   = CommsVerifier()

    executor = MultiThreadedExecutor()
    executor.add_node(pub_robot)
    executor.add_node(pub_action)
    executor.add_node(verifier)

    # 在后台线程里 spin executor
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # 等待测试完成
    print(f"\n运行中，等待 {TEST_DURATION_SEC} 秒...\n")
    time.sleep(TEST_DURATION_SEC)

    # 打印结果
    print("\n" + "=" * 60)
    expected = int(TEST_DURATION_SEC * PUBLISH_HZ)
    print(f"  测试结果（期望每个话题约 {expected} 条）：")
    print(f"  robot_state  收到: {verifier.robot_count:3d} 条  ", end="")
    print("✅" if verifier.robot_count >= expected * 0.8 else "❌ 偏少")
    print(f"  object_pose  收到: {verifier.object_count:3d} 条  ", end="")
    print("✅" if verifier.object_count >= expected * 0.8 else "❌ 偏少")
    print(f"  action_cmd   收到: {verifier.action_count:3d} 条  ", end="")
    print("✅" if verifier.action_count >= expected * 0.8 else "❌ 偏少")

    total_ok = (
        verifier.robot_count  >= expected * 0.8 and
        verifier.object_count >= expected * 0.8 and
        verifier.action_count >= expected * 0.8
    )
    print()
    if total_ok:
        print("  ✅ 测试通过！ROS2 pub/sub 通路在本机完全正常。")
        print("  → 真机上同样的话题名/消息类型可以直接使用。")
    else:
        print("  ❌ 测试未通过，请检查环境。")
    print("=" * 60)

    executor.shutdown(timeout_sec=1.0)
    rclpy.shutdown()
    return 0 if total_ok else 1


if __name__ == "__main__":
    sys.exit(main())
