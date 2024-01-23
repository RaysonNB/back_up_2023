#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.pytorch_models import *
from pcms.openvino_models import *
import numpy as np
from geometry_msgs.msg import Twist
import math
from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest
import time
from std_msgs.msg import String
from rospkg import RosPack
from tf.transformations import euler_from_quaternion
from typing import Tuple, List

def callback_image1(msg):
    global _image1
    _image1 = CvBridge().imgmsg_to_cv2(msg, "bgr8")


def callback_depth1(msg):
    global _depth1
    _depth1 = CvBridge().imgmsg_to_cv2(msg, "passthrough")


def get_real_xyz(x, y):
    global depth
    a = 49.5 * np.pi / 180
    b = 60.0 * np.pi / 180
    d = depth[y][x]
    h, w = depth.shape[:2]
    if d == 0:
        for k in range(1, 15, 1):
            if d == 0 and y - k >= 0:
                for j in range(x - k, x + k, 1):
                    if not (0 <= j < w): continue
                    d = depth[y - k][j]
                    if d > 0: break
            if d == 0 and x + k < w:
                for i in range(y - k, y + k, 1):
                    if not (0 <= i < h): continue
                    d = depth[i][x + k]
                    if d > 0: break
            if d == 0 and y + k < h:
                for j in range(x + k, x - k, -1):
                    if not (0 <= j < w): continue
                    d = depth[y + k][j]
                    if d > 0: break
            if d == 0 and x - k >= 0:
                for i in range(y + k, y - k, -1):
                    if not (0 <= i < h): continue
                    d = depth[i][x - k]
                    if d > 0: break
            if d > 0: break

    x = int(x) - int(w // 2)
    y = int(y) - int(h // 2)
    real_y = round(y * 2 * d * np.tan(a / 2) / h)
    real_x = round(x * 2 * d * np.tan(b / 2) / w)
    #print(d)
    return real_x, real_y, d


def get_pose_target(pose, num):
    p = []
    for i in [num]:
        if pose[i][2] > 0:
            p.append(pose[i])

    if len(p) == 0: return -1, -1
    return int(p[0][0]), int(p[0][1])


def get_distance(px, py, pz, ax, ay, az, bx, by, bz):
    A, B, C, p1, p2, p3, qx, qy, qz, distance = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    A = int(bx) - int(ax)
    B = int(by) - int(ay)
    C = int(bz) - int(az)
    p1 = int(A) * int(px) + int(B) * int(py) + int(C) * int(pz)
    p2 = int(A) * int(ax) + int(B) * int(ay) + int(C) * int(az)
    p3 = int(A) * int(A) + int(B) * int(B) + int(C) * int(C)
    if (p1 - p2) != 0 and p3 != 0:
        t = (int(p1) - int(p2)) / int(p3)
        qx = int(A) * int(t) + int(ax)
        qy = int(B) * int(t) + int(ay)
        qz = int(C) * int(t) + int(az)
        distance = int(pow(((int(px) - int(qx)) ** 2 + (int(py) - int(qy)) ** 2 + (int(pz) - int(qz)) ** 2), 0.5))
        return int(distance)
    return 0


def pose_draw(img):
    cx7, cy7, cx9, cy9, cx5, cy5 = 0, 0, 0, 0, 0, 0
    global ax, ay, az, bx, by, bz
    n1, n2, n3 = 6, 8, 10
    cx7, cy7 = get_pose_target(pose, n2)

    cx9, cy9 = get_pose_target(pose, n3)

    cx5, cy5 = get_pose_target(pose, n1)
    if cx7 == -1 and cx9 != -1:
        cv2.circle(img, (cx5, cy5), 5, (0, 255, 0), -1)
        ax, ay, az = get_real_xyz(cx5, cy5)

        cv2.circle(img, (cx9, cy9), 5, (0, 255, 0), -1)
        bx, by, bz = get_real_xyz(cx9, cy9)
    elif cx7 != -1 and cx9 == -1:

        cv2.circle(img, (cx5, cy5), 5, (0, 255, 0), -1)
        ax, ay, az = get_real_xyz(cx5, cy5)

        cv2.circle(img, (cx7, cy7), 5, (0, 255, 0), -1)
        bx, by, bz = get_real_xyz(cx7, cy7)
    elif cx7 == -1 and cx9 == -1:
        pass
    else:
        cv2.circle(img, (cx7, cy7), 5, (0, 255, 0), -1)
        ax, ay, az = get_real_xyz(cx7, cy7)

        cv2.circle(img, (cx9, cy9), 5, (0, 255, 0), -1)
        bx, by, bz = get_real_xyz(cx9, cy9)
    return img

def callback_image(msg):
    global _frame
    _frame = cv2.flip(CvBridge().imgmsg_to_cv2(msg, "bgr8"), 0)


    
def move(forward_speed: float = 0, turn_speed: float = 0):
    global _cmd_vel
    msg = Twist()
    msg.linear.x = forward_speed
    msg.angular.z = turn_speed
    _cmd_vel.publish(msg)
if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    d = 1
    
    t = 0
    s=""
    
    one = 0
    _image1 = None
    _topic_image1 = "/cam1/rgb/image_raw"
    rospy.Subscriber(_topic_image1, Image, callback_image1)
    _depth1 = None
    _topic_depth1 = "/cam1/depth/image_raw"
    rospy.Subscriber(_topic_depth1, Image, callback_depth1)

    _cmd = "point"
    dnn_rcnn = Yolov8("yolov8n")
    net_pose = HumanPoseEstimation()
    ax, ay, az, bx, by, bz ,cx,cy= 0, 0, 0, 0, 0, 0,0,0
    min1 = 99999999
    cnt=0
    while not rospy.is_shutdown():
        
        rospy.Rate(20).sleep()
        if _image1 is None: 
            continue
            cnt+=1
        
        if _depth1 is None: 
            continue
            cnt+=1
        fcnt = 1
        img = _image1.copy()
        depth = _depth1.copy()
        
        pose = None
        poses = net_pose.forward(img)
        
        if _cmd == "point":
            detections = dnn_rcnn.forward(img)[0]["det"]
            for i, detection in enumerate(detections):
                x1, y1, x2, y2, score, class_id = map(int, detection)
                print(class_id)
                score = detection[4]
                if class_id != 24 and class_id != 26 and class_id != 28: continue
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                px, py, pz = get_real_xyz(cx, cy)
                cnt = get_distance(px, py, pz, ax, ay, az, bx, by, bz)
            
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
            for i, pose in enumerate(poses):
                point = []
                for j, (x, y, preds) in enumerate(pose):
                    if preds <= 0: continue
                    x, y = map(int, [x, y])
                    for num in [8, 10]:
                        point.append(j)
                if len(point) == 2:
                    pose = poses[i]
                    break

            flag = None
            if pose is not None:
                img=pose_draw(img)
                for i, detection in enumerate(detections):
                    x1, y1, x2, y2, score, class_id = map(int, detection)
                    score = detection[4]
                    if (class_id == 24 or class_id == 26 or class_id == 28):

                        cx1 = (x2 - x1) // 2 + x1
                        cy1 = (y2 - y1) // 2 + y1

                        px, py, pz = get_real_xyz(cx1, cy1)
                        cnt = get_distance(px, py, pz, ax, ay, az, bx, by, bz)

                        cnt = int(cnt)
                        #print(cnt)
                        if cnt < min1 and cnt < 600 and cnt != 0:
                            flag = i
                for i, detection in enumerate(detections):
                    #print("1")
                    x1, y1, x2, y2, score, class_id = map(int, detection)
                    if (class_id == 24 or class_id == 26 or class_id == 28):
                        if i == flag:
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 5)
                            break
        '''
        elif _cmd == "bring":
            rx, ry, rz = get_real_xyz(cx, cy)
            if(rz==0): 
                _voice = None
                continue

            angle = np.arctan2(rx, rz)
            msg.angular.z=angle
            _cmd_vel.publish(msg)
            time.sleep(5)
            msg.angular.z=0
            cx, cy = w // 2, h // 2
            for i in range(cy + 1, h):
                if depth[cy][cx] == 0 or 0 < depth[i][cx] < depth[cy][cx]:
                    cy = i 
            _,_,d = get_real_xyz(cx,cy)
            while d > 0 or abs(e) >= 1:
                _,_,d1 = get_real_xyz(cx,cy)
                e = d1 - 550
                if abs(e)<=1:
                    break
                v = 0.001 * e
                if v > 0:
                    v = min(v, 0.3)
                if v < 0:
                    v = max(v, -0.3)
                #print(d1, e, v)
                
                move(v, 0)
            step = "wait"
            _cmd = ""
        if step == "wait":
            publisher_speaker.publish("can you give me the bag")
            joint1, joint2, joint3, joint4 = 0.000, -1.0, 0.3,0.70
            set_joints(joint1, joint2, joint3, joint4, t)
            time.sleep(t)
            open_gripper(t)
            step="wait2"
        if step == "wait2":
            s=s.lower()
            if s!="":
                print(s)
            if "ok" in s or "put" in s or "Put" in s or "PUT" in s or "OK" in s or "Ok" in s:
                publisher_speaker.publish("I have already got it")
                close_gripper(t)
                time.sleep(t)
        '''
        cv2.imshow("image", img)
        cv2.waitKey(16)
        #print(cnt)
    rospy.loginfo("demo3 end!")
