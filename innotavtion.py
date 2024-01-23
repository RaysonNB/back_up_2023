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
    
def point():
    for id, index, preds, x1, y1, x2, y2 in boxes:
        if index != 0:
            continue
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        px,py,pz=get_real_xyz(cx, cy)
        cnt=get_distance(px,py,pz,ax,ay,az,bx,by,bz)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
        cv2.putText(frame, str(int(cnt)//10), (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
    for i, pose in enumerate(poses):
        point = []
        for j, (x,y,preds) in enumerate(pose): #x: ipex 坐標 y: ipex 坐標 preds: 准度
            if preds <= 0: continue
            x,y = map(int,[x,y])
            for num in [8,10]:
                point.append(j)
        if len(point) == 2:
            pose = poses[i]
            break

    flag=None
    if pose is not None:
        pose_draw()
        for id, index, conf, x1, y1, x2, y2 in boxes:
            if(index == 39):

                cx1 = (x2 - x1) // 2 + x1
                cy1 = (y2 - y1) // 2 + y1

                px,py,pz=get_real_xyz(cx1, cy1)
                cnt=get_distance(px,py,pz,ax,ay,az,bx,by,bz)
                
                cnt=int(cnt)
                if cnt< min1 and cnt<400 and cnt != 0:
                    flag = id
        for id, index, conf, x1, y1, x2, y2 in boxes:
            if(index == 39):
                if id == flag:
                    cv2.putText(frame, str(int(cnt)//10), (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    break
        
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
        
def open_gripper(t):
    return set_gripper(0.01, t)

def close_gripper(t):
    return set_gripper(-0.01, t)

if __name__  == "__main__":

    rospy.init_node("demo3")
    rospy.loginfo("demo3 start!")

    _cmds = None
    image = None
    _topic_image1 = "/cam1/rgb/image_raw"
    rospy.Subscriber(_topic_image1, Image, callback_image1)
    
    # Depth Image Subscriber
    _depth1 = Nones
    _topic_depth1 = "/cam1/depth/image_raw"
    rospy.Subscriber(_topic_depth1, Image, callback_depth1)

    _frame = None
    rospy.Subscriber("/cam2/rgb/image_raw", Image, callback_image)

    _depth= None
    rospy.Subscriber("/cam2/depth/image_raw", Image, callback_depth)

    _cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    
    rospy.Subscriber("/voice/text", Voice, callback_voice)
    
    publisher_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)
    topic_imu = "/imu/data"
    _imu=None
    rospy.Subscriber(topic_imu, Imu, callback_imu)
    rospy.wait_for_message(topic_imu, Imu)
    with open("/home/pcms/catkin_ws/src/beginner_tutorials/src/sprit_QandA.txt", "r") as f:
        try:
            _cmds = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
    
    
    _voice = None
    _topic_voice = "/voice/text"
    rospy.Subscriber(_topic_voice, Voice, callback_voice)

    # arm_data = [-0.003, -1.05, 0.354, 0.706]

    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        if image is None:continue
        frame = image.copy()
        cnt = 1
        frame = _frame.copy()
        depth=_depth.copy()
        
        min1=99999999
        
        boxes = ddn_rcnn.forward(frame)
        
        pose=None
        poses = net_pose.forward(frame)
        if _voice is not None:
            rospy.loginfo("%s (%d)" % (_voice.text, _voice.direction))
            _cmd = text_to_cmd(_voice.text)
        if _cmd == "point":
            point()
            publisher_speaker.publish("That is a sandwich, are you going to order it")
        else if _cmd == "bring":
            #chasses
            open_gripper(3)
            set_joints(-0.176,-0,410,-0.733,0.884, 3)
            close_gripper(3)
            set_joints(-0.003, -1.05, 0.354, 0.706, 3)
            #chasses
        else if _cmd == "hello":
            point()
        else if "want" in _cmd and "this" in _cmd :
            #walk_death
        else if "bottle" in _cmd:
            point()
            #walk_death
            set_joints(-0.003, -1.05, 0.354, 0.706)
            open_gripper(3)
        else:
            publisher_speaker.publish(_cmd)
        cv2.imshow("image",image)       
    rospy.loginfo("demo3 end!")

