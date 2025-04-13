#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import dompc_api
import acados_api
from ecn_mpc.splines import Splines
import numpy as np

from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
from sensor_msgs.msg import JointState
from math import atan2, cos, sin

Tmax = 5.


class Control(Node):
    def __init__(self):

        super().__init__('mpc', namespace='zoe')

        dt = 0.1

        self.splines = Splines()

        if self.declare_parameter('dompc', True).value:
            self.solver = dompc_api.ZoeMPC(dt)
        else:
            self.solver = acados_api.ZoeMPC(dt)

        # init plumbing
        self.cmd = AckermannDrive()
        self.cmd_pub = self.create_publisher(AckermannDrive, 'cmd', 1)
        self.steering = None
        self.js_sub = self.create_subscription(JointState, 'joint_states', self.js_cb, 1)
        self.tf_buffer = Buffer(node=self)
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.path_sub = self.create_subscription(Path, 'plan', self.path_cb, 1)

        self.spline_path_pub = self.create_publisher(Path, 'local_plan', 1)
        self.spline_path = Path()
        self.spline_path.header.frame_id = 'map'

        self.mpc_path_pub = self.create_publisher(Path, 'predicted_plan', 1)
        self.mpc_path = Path()
        self.mpc_path.header.frame_id = 'map'

        self.create_timer(dt, self.move)

    def js_cb(self, js: JointState):
        if 'steering' not in js.name:
            return
        self.steering = js.position[js.name.index('steering')]

    def path_cb(self, path: Path):
        self.splines.set_path(path)

    def pose(self):
        now = rclpy.time.Time()
        if not self.tf_buffer.can_transform('map', 'zoe/base_link', now):
            return [None]*3
        pose = self.tf_buffer.lookup_transform('map', 'zoe/base_link', now).transform

        return pose.translation.x, pose.translation.y, 2*atan2(pose.rotation.z,pose.rotation.w)

    def publish_ref(self, ref):
        '''
        publishes output of spline fitting on Tmax horizon for reference
        '''
        self.spline_path.poses = []
        self.spline_path.header.stamp = self.get_clock().now().to_msg()

        for t in np.linspace(0, Tmax, 20):
            wp = ref(t)
            pose = PoseStamped()
            pose.pose.position.x = wp[0]
            pose.pose.position.y = wp[1]
            pose.pose.position.z = 0.2
            self.spline_path.poses.append(pose)
        self.spline_path_pub.publish(self.spline_path)

    def move(self):

        x,y,theta = self.pose()

        if self.steering is None or x is None:
            return

        ref = self.splines.spline_from(x,y,Tmax)

        if ref is None:
            return
        self.publish_ref(ref)

        # call MPC from the current state
        u, traj = self.solver.solve([x,y,theta,self.steering], ref)

        self.cmd.speed = u[0]
        self.cmd.steering_angle_velocity = u[1]

        self.cmd_pub.publish(self.cmd)

        # also display predicted trajectory
        self.mpc_path.poses = []
        for x,y,theta,_ in traj:
            pose = PoseStamped()
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.2
            pose.pose.orientation.z = sin(theta/2.)
            pose.pose.orientation.w = cos(theta/2.)
            self.mpc_path.poses.append(pose)
        self.mpc_path.header.stamp = self.get_clock().now().to_msg()
        self.mpc_path_pub.publish(self.mpc_path)


rclpy.init()
rclpy.spin(Control())
rclpy.shutdown()

