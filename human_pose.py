#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import Yolov8,HumanPoseEstimation
import numpy as np
import subprocess
#gemini2
def callback_image2(msg):
    global frame2
    frame2 = CvBridge().imgmsg_to_cv2(msg, "bgr8")

def callback_depth2(msg):
    global depth2
    depth2 = CvBridge().imgmsg_to_cv2(msg, "passthrough")
def get_real_xyz(dp,x, y):
    a = 55.0 * np.pi / 180
    b = 86.0 * np.pi / 180
    d = dp[y][x]
    h, w = dp.shape[:2]
    x = int(x) - int(w // 2)
    y = int(y) - int(h // 2)
    real_y = round(y * 2 * d * np.tan(a / 2) / h)
    real_x = round(x * 2 * d * np.tan(b / 2) / w)
    return real_x, real_y, d
if __name__ == "__main__":
    rospy.init_node("human_pose")
    
    
    frame2 = None
    rospy.Subscriber("/camera/color/image_raw", Image, callback_image2)

    depth2 = None
    rospy.Subscriber("/camera/depth/image_raw", Image, callback_depth2)
    
    print("yolo")
    net_pose = HumanPoseEstimation(device_name="GPU")
    
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        if frame2 is None: 
            print("frame2")
            continue
        if depth2 is None: 
            print("depth2")
            continue
        t_pose=None
        points=[]
        A=[]
        B=[]
        poses = net_pose.forward(frame2)
        if len(poses) > 0:
            YN=-1
            a_num,b_num=10,8
            if poses[0][10][2] > 0 and poses[0][8][2] > 0:
                YN=0
                a_num,b_num=10,8
            elif poses[0][9][2] > 0 and poses[0][7][2] > 0:
                YN=0
                a_num,b_num=9,7
            
            A = list(map(int, poses[0][a_num][:2]))
            ax,ay,az=get_real_xyz(depth2,A[0],A[1])
            B = list(map(int, poses[0][b_num][:2]))
            bx,by,bz=get_real_xyz(depth2,B[0],B[1])
        print(A,B)
        if len(A) !=0:
            cv2.circle(frame2, (A[0],A[1]), 3, (0, 255, 0), -1)
        if len(B) !=0:
            cv2.circle(frame2, (B[0],B[1]), 3, (0, 255, 0), -1)
        cv2.imshow("capture", frame2)
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            break
    
