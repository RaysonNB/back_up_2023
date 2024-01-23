#!/usr/bin/env python3.8
import rospy
from geometry_msgs.msg import Twist


if __name__ == "__main__":
    rospy.init_node("move_demo")
    rospy.loginfo("started")
    msg_cmd = Twist()
    pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

    msg_cmd.linear.x = 0.15
    pub_cmd.publish(msg_cmd)
    msg_cmd.linear.x = 0.0
    pub_cmd.publish(msg_cmd)
    rospy.loginfo("end")
    
