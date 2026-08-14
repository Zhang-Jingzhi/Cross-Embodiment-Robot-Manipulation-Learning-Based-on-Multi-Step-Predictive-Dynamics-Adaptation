"""
Franka ROS2 Interface Node
===========================
真机上运行的 ROS2 桥接节点。
负责：
  1. 订阅 franka_ros2 发布的机器人关节/末端状态
  2. 订阅感知节点发布的物体位姿
  3. 接收 deploy_franka.py 发来的 action，转发给 fr3_controller_lin.py

运行方式（真机 Ubuntu 22.04，source franka_ros2 环境后）:
    conda activate 0320
    python franka_ros2_interface.py

⚠️  需要先在另一个终端启动 franka_ros2 的控制器:
    ros2 launch franka_bringup franka.launch.py robot_ip:=<IP>
"""

import rclpy
from rclpy.node import Node

import numpy as np
import threading

from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray


# ── ROS2 Topic 名（与 fr3_controller_lin.py 对齐）────────────────────────────
# 根据你学长的控制器实际 topic 名调整
EEF_STATE_TOPIC    = "/franka/end_effector_pose"     # 末端位姿 (PoseStamped)
JOINT_STATE_TOPIC  = "/franka/joint_states"           # 关节状态 (JointState)
OBJECT_POSE_TOPIC  = "/object_pose"                   # 感知节点输出 (PoseStamped)
DELTA_CMD_TOPIC    = "/cartesian_delta_cmd"            # 接收 action (Twist)
GRIPPER_CMD_TOPIC  = "/franka/gripper_cmd"             # 夹爪指令 (Float32MultiArray)


class FrankaROS2Interface(Node):
    """
    ROS2 节点：在 deploy_franka.py 的 FrankaStateProvider 和 FrankaActionExecutor
    中被调用。也可以作为独立节点运行，供调试使用。
    """

    def __init__(self):
        super().__init__("pace_franka_interface")
        self._lock = threading.Lock()

        # ── 状态缓存 ───────────────────────────────────────────────────────────
        self.eef_pos        = np.zeros(3, dtype=np.float32)
        self.eef_quat       = np.array([0., 0., 0., 1.], dtype=np.float32)  # xyzw
        self.gripper_pos    = np.zeros(2, dtype=np.float32)  # [finger1, finger2] (m)
        self.object_pos     = np.zeros(3, dtype=np.float32)
        self.goal_pos       = np.zeros(3, dtype=np.float32)

        # ── Subscribers ────────────────────────────────────────────────────────
        self.create_subscription(
            PoseStamped, EEF_STATE_TOPIC,
            self._eef_callback, 10
        )
        self.create_subscription(
            JointState, JOINT_STATE_TOPIC,
            self._joint_state_callback, 10
        )
        self.create_subscription(
            PoseStamped, OBJECT_POSE_TOPIC,
            self._object_pose_callback, 10
        )

        # ── Publishers ─────────────────────────────────────────────────────────
        self.delta_pub   = self.create_publisher(Twist, DELTA_CMD_TOPIC, 10)
        self.gripper_pub = self.create_publisher(Float32MultiArray, GRIPPER_CMD_TOPIC, 10)

        self.get_logger().info("FrankaROS2Interface ready.")

    # ──────────────────────────────────────────────────────────────────────────
    # Subscriber Callbacks
    # ──────────────────────────────────────────────────────────────────────────

    def _eef_callback(self, msg: PoseStamped):
        with self._lock:
            self.eef_pos[0] = msg.pose.position.x
            self.eef_pos[1] = msg.pose.position.y
            self.eef_pos[2] = msg.pose.position.z
            self.eef_quat[0] = msg.pose.orientation.x
            self.eef_quat[1] = msg.pose.orientation.y
            self.eef_quat[2] = msg.pose.orientation.z
            self.eef_quat[3] = msg.pose.orientation.w

    def _joint_state_callback(self, msg: JointState):
        """
        Franka 的 panda_finger_joint1 和 panda_finger_joint2 是夹爪关节。
        开合范围: 0 (闭合) ~ 0.04 (完全张开)，单位 m。
        """
        with self._lock:
            for i, name in enumerate(msg.name):
                if "finger_joint1" in name:
                    self.gripper_pos[0] = msg.position[i]
                elif "finger_joint2" in name:
                    self.gripper_pos[1] = msg.position[i]

    def _object_pose_callback(self, msg: PoseStamped):
        """
        感知节点（ArUco / FoundationPose / Realsense 等）输出的物体位置。
        确保坐标系已通过手眼标定转换到机器人基座坐标系。
        """
        with self._lock:
            self.object_pos[0] = msg.pose.position.x
            self.object_pos[1] = msg.pose.position.y
            self.object_pos[2] = msg.pose.position.z

    # ──────────────────────────────────────────────────────────────────────────
    # 状态读取接口（供 FrankaStateProvider 调用）
    # ──────────────────────────────────────────────────────────────────────────

    def get_eef_pos(self) -> np.ndarray:
        with self._lock:
            return self.eef_pos.copy()

    def get_gripper_pos(self) -> np.ndarray:
        with self._lock:
            return self.gripper_pos.copy()

    def get_object_pos(self) -> np.ndarray:
        with self._lock:
            return self.object_pos.copy()

    # ──────────────────────────────────────────────────────────────────────────
    # Action 发送接口（供 FrankaActionExecutor 调用）
    # ──────────────────────────────────────────────────────────────────────────

    def send_cartesian_delta(self, delta_xyz: np.ndarray, gripper_open: bool):
        """
        发送末端位移指令给 fr3_controller_lin.py。

        fr3_controller_lin.py 期望的消息格式：
            Twist.linear.{x,y,z} = 末端期望位移 (m)
            Twist.angular.z      = 夹爪：+1.0=张开, -1.0=闭合
        （如果你学长的控制器用不同格式，在这里修改）
        """
        msg = Twist()
        msg.linear.x  = float(delta_xyz[0])
        msg.linear.y  = float(delta_xyz[1])
        msg.linear.z  = float(delta_xyz[2])
        msg.angular.z = 1.0 if gripper_open else -1.0
        self.delta_pub.publish(msg)

    def send_gripper_cmd(self, width: float, speed: float = 0.1, force: float = 10.0):
        """
        直接控制夹爪开合宽度。
        width: 目标开合宽度 (m)，0=全闭，0.08=全开 (FR3)
        """
        msg = Float32MultiArray()
        msg.data = [width, speed, force]
        self.gripper_pub.publish(msg)


# ── 独立运行（调试用）─────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = FrankaROS2Interface()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
