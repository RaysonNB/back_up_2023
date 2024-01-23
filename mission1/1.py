#!/usr/bin/env python3
import rospy
import time
from std_msgs.msg import String


if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("start")
    
    pub = rospy.Publisher("/speaker/say", String, queue_size=10)
    
    time.sleep(1)
    pub.publish("你好，我是校园5G助手，请出示新生晚会邀请码")
    time.sleep(2)

