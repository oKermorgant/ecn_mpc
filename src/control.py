#!/usr/bin/env python3

import rclpy
from rclpy.node import Node


class Control(Node):
    def __init__(self):

        super().__init__('mpc')









rclpy.init()
rclpy.spin(Control())
rclpy.shutdown()

