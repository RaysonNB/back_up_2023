#!/usr/bin/env python3.8
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu

from cv_bridge import CvBridge
import cv2

from geometry_msgs.msg import Twist
from RobotChassis import RobotChassis

if __name__ == "__main__":
    rospy.init_node("rviz_demo")
    rospy.loginfo("rviz_demo task")
    chassis = RobotChassis()
    msg_cmd = Twist()
    pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    fablab = {"door":(9.78,4.52,3.14),"seat":(11.8,3.92,1.57),"seat1":(12.3,4.98,1.57),"seat2":(1.5,5,1.57),"roomM": (5,8.78,3.05),"m_to_d":(7.02,-2.21,1.57),"D_TO_S":(5.42,-2.06,0)}
    RCJPos = {"roomL" : (4.62,9.19,-0.75),"roomR" : (4.62,9.19,1.2),"roomM": (4.62,9.19,-2.2),"master" : (7.79,11.1,2.4),"come_wait" :(7.39,8.54,3),"master_wait" :(6.5,8.54,0),"CL" :(3.2,10.5,1.05),"CR":(5.38,11.6,3.14)}   
    test = {"door":(0.143,2.42,3.14),"seat":(11.8,3.92,1.57),"seat1":(12.3,4.98,1.57),"seat2":(1.5,5,1.57),"roomM": (5,8.78,3.05),"m_to_d":(7.02,-2.21,1.57),"D_TO_S":(5.42,-2.06,0)}

    #for i in fablab.keys():
     #   chassis.move_to(*fablab[i])
    #    print("arrive the %s" % i)
    #chassis.move_to(*fablab["roomM"])
    chassis.move_to(-5.24,-2.71,3.14)
    #chassis.move_to(*task3["door"])
    #chassis.move_to(*pos["master"])
rospy.loginfo("end")
