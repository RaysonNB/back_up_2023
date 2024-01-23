#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import Twist

def move_robot(forward_speed, turn_speed):
    global _pub_cmd
    msg = Twist()
    msg.linear.x = forward_speed
    msg.angular.z = turn_speed
    _pub_cmd.publish(msg)

# 定義直行距離函數
def move_distance(distance):
    # 將距離單位轉換為米
    distance = distance / 1000.0
    # 設定直行速度和時間，假設直行速度為0.2米/秒
    forward_speed = 0.2
    forward_time = abs(distance) / forward_speed
    # 開始直行
    move_robot(forward_speed, 0)
    rospy.sleep(forward_time)
    # 停止直行
    move_robot(0, 0)

# 呼叫直行距離函數，假設距離為1000毫米
if __name__ == "__main__":
    rospy.init_node('test_move_distance')
    _msg_cmd = Twist()
    _pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    
    move_distance(1000)
