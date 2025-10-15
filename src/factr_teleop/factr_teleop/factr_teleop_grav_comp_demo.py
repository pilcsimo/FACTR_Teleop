# ---------------------------------------------------------------------------
# FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning
# https://arxiv.org/abs/2502.17432
# Copyright (c) 2025 Jason Jingzhou Liu and Yulong Li

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------


import time
# ...existing code...
import numpy as np
import rclpy

from factr_teleop.factr_teleop import FACTRTeleop
from rcl_interfaces.msg import SetParametersResult

# ZMQ bridge
try:
    from python_utils.zmq_messenger import ZMQPublisher, ZMQSubscriber
except Exception:
    ZMQPublisher = None
    ZMQSubscriber = None

class FACTRTeleopGravComp(FACTRTeleop):
    """
    Gravity-comp leader-only demo that also bridges ZMQ:
    - Publishes leader joint action [7 arm, 1 gripper] to a ZMQ PUB.
    - Optionally subscribes to a ZMQ SUB and forwards that action instead (when configured).
    """

    def __init__(self):
        super().__init__()

    def set_up_communication(self):
        # ZMQ config (optional)
        zmq_cfg = self.config.get("zmq", {})
        self._zmq_action_pub_addr = zmq_cfg.get("action_pub", "tcp://127.0.0.1:6001")
        self._zmq_action_sub_addr = zmq_cfg.get("action_sub", None)
        self._zmq_forward_source = zmq_cfg.get("forward_source", "device")  # 'device' or 'zmq'

        # Allow runtime override
        self.declare_parameter("zmq.forward_source", self._zmq_forward_source)
        self._zmq_forward_source = self.get_parameter("zmq.forward_source").get_parameter_value().string_value

        # Create endpoints if library available
        if ZMQPublisher is None:
            self.get_logger().warn("ZMQPublisher/ZMQSubscriber not available. ZMQ bridge disabled.")
            self._zmq_pub = None
            self._zmq_sub = None
            return

        self._zmq_pub = ZMQPublisher(self._zmq_action_pub_addr)
        self._zmq_sub = ZMQSubscriber(self._zmq_action_sub_addr) if self._zmq_action_sub_addr else None

        self.get_logger().info(
            f"ZMQ bridge: publish -> {self._zmq_action_pub_addr}; "
            f"subscribe <- {self._zmq_action_sub_addr or 'None'}; "
            f"forward_source={self._zmq_forward_source}"
        )

        # Update forward_source dynamically
        self.add_on_set_parameters_callback(self._on_param_update)

    def _on_param_update(self, params):
        """
        Validate/consume zmq.forward_source param updates.
        Must return SetParametersResult for rclpy.
        """
        res = SetParametersResult(successful=True, reason="ok")
        for p in params:
            if p.name == "zmq.forward_source":
                val = str(p.value).lower()
                if val not in ("device", "zmq"):
                    res.successful = False
                    res.reason = "zmq.forward_source must be 'device' or 'zmq'"
                else:
                    self._zmq_forward_source = val
                    self.get_logger().info(f"Updated zmq.forward_source = {self._zmq_forward_source}")
        return res

    def get_leader_gripper_feedback(self):
        # No follower gripper feedback in this demo
        return 0.0
    
    def gripper_feedback(self, leader_gripper_pos, leader_gripper_vel, gripper_feedback):
        # No gripper haptics in this demo
        return 0.0
    
    def get_leader_arm_external_joint_torque(self):
        # No follower torque feedback in this demo
        return np.zeros(self.num_arm_joints, dtype=float)

    def update_communication(self, leader_arm_pos, leader_gripper_pos):
        """
        Called each control tick. Publish either:
        - device action [q(7), gripper] (forward_source='device'), or
        - latest ZMQ action (forward_source='zmq') if available.
        """
        if self._zmq_pub is None:
            return

        publish_vec = None

        if self._zmq_forward_source == "zmq" and self._zmq_sub is not None and self._zmq_sub.message is not None:
            arr = np.asarray(self._zmq_sub.message, dtype=float).ravel()
            if arr.size >= 8:
                publish_vec = arr[:8]
            elif arr.size == 7:
                publish_vec = np.concatenate([arr[:7], [0.0]])
            # if malformed, fall back to device
        if publish_vec is None:
            # Use current device state
            publish_vec = np.concatenate([leader_arm_pos.astype(float), [float(leader_gripper_pos)]])

        try:
            self._zmq_pub.send_message(publish_vec)
        except Exception as e:
            self.get_logger().warn(f"ZMQ publish failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    factr_teleop_grav_comp = FACTRTeleopGravComp()

    try:
        while rclpy.ok():
            rclpy.spin(factr_teleop_grav_comp)
    except KeyboardInterrupt:
        factr_teleop_grav_comp.get_logger().info("Keyboard interrupt received. Shutting down...")
        factr_teleop_grav_comp.shut_down()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()