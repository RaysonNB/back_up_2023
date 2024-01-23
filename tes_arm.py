#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.pytorch_models import *
from pcms.openvino_models import Yolov8, HumanPoseEstimation
import numpy as np
from geometry_msgs.msg import Twist
import math
from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest
from open_manipulator_msgs.srv import SetKinematicsPose, SetKinematicsPoseRequest
import time
from mr_voice.msg import Voice
from std_msgs.msg import String
from rospkg import RosPack
from follow import FollowMe
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import Imu
from typing import Tuple, List
from RobotChassis import RobotChassis
def say(a):
    global publisher_speaker
    publisher_speaker.publish(a)
def set_joints(joint1, joint2, joint3, joint4, t):
    service_name = "/goal_joint_space_path"
    rospy.wait_for_service(service_name)

    try:
        service = rospy.ServiceProxy(service_name, SetJointPosition)

        request = SetJointPositionRequest()
        request.joint_position.joint_name = [
            "joint1", "joint2", "joint3", "joint4"]
        request.joint_position.position = [joint1, joint2, joint3, joint4]
        request.path_time = t

        response = service(request)
        return response
    except Exception as e:
        rospy.loginfo("%s" % e)
        return False
def set_gripper(angle, t):
    service_name = "/goal_tool_control"
    rospy.wait_for_service(service_name)
    
    try:
        service = rospy.ServiceProxy(service_name, SetJointPosition)
        
        request = SetJointPositionRequest()
        request.joint_position.joint_name = ["gripper"]
        request.joint_position.position = [angle]
        request.path_time = t
        
        response = service(request)
        return response
    except Exception as e:
        rospy.loginfo("%s" % e)
        return False

def open_gripper(t):
    return set_gripper(0.01, t)

def close_gripper(t):
    return set_gripper(-0.01, t)
    
def callback_voice(msg):
    global s
    s = msg.text
if __name__ == "__main__": 
   
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    publisher_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)
    yes="1"
    s=""
    _cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    step="no"
    rospy.Subscriber("/voice/text", Voice, callback_voice)
    t=3.0
    while not rospy.is_shutdown():
        msg=Twist()
        s=s.lower()
        if step !="wait":
            if yes=="1":
                t=0.3
                open_gripper(t)
                time.sleep(t)
                t=3.0
                joint1, joint2, joint3, joint4 = 0.000, 0.75, 1.0,-1.0
                set_joints(joint1, joint2, joint3, joint4, t)
                time.sleep(t)
                for i in range(70):
                    msg.linear.x=0.2
                    _cmd_vel.publish(msg)
                close_gripper(t)
                break
            else:
                time.sleep(3.0)
                publisher_speaker.publish("can you give me the bag")
                joint1, joint2, joint3, joint4 = 0.000, -1.0, 0.3,0.70
                set_joints(joint1, joint2, joint3, joint4, t)
                time.sleep(t)
                open_gripper(t)
                step="wait"
        else:
            if "ok" in s or "put" in s:
                publisher_speaker.publish("I have already got it, I will follow you now")
                close_gripper(t)
                time.sleep(t)
                break
        
