#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import Yolov8, HumanPoseEstimation
import numpy as np
from geometry_msgs.msg import Twist
from pcms.pytorch_models import *
import math

#https://github.com/supercatex/code/blob/master/ros/demo3.py
def callback_image(msg):
    global _frame
    _frame = CvBridge().imgmsg_to_cv2(msg, "bgr8")
def say(a): 
    publisher_speaker.publish(a) 

def callback_depth(msg):
    global _depth
    _depth = CvBridge().imgmsg_to_cv2(msg, "passthrough")
    
def get_real_xyz(x, y):
    global _depth
    a = 49.5 * np.pi / 180
    b = 60.0 * np.pi / 180
    d = _depth[y][x]
    h, w = _depth.shape[:2]
    x = x - w // 2
    y = y - h // 2
    real_y = y * 2 * d * np.tan(a / 2) / h
    real_x = x * 2 * d * np.tan(b / 2) / w
    return real_x, real_y, d
    
def get_pose_target(pose,num):
    p = []
    for i in [num]:
        if pose[i][2] > 0:
            p.append(pose[i])
    
    if len(p) == 0: return -1, -1
    return int(p[0][0]),int(p[0][1])
    
'''
def find_all():
    global ddn_rcnn
    global frame
    global boxes
    boxes = ddn_rcnn.forward(frame)
    if len(boxes) == 0:
        return "nothing"
    for id, index, conf, x1, y1, x2, y2 in boxes:
        name=ddn_rcnn.labels[index]
        #if namee=="bottle": #name=="suitcase" or name=="backpack":
        cv2.putText(frame, name, (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cx = (x2 - x1) // 2 + x1
        cy = (y2 - y1) // 2 + y1
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        
        return (x1, y1), (x2, y2), (cx, cy), name
'''

def get_target(poses):
    target = -1
    target_d = 9999999
    for i, pose in enumerate(poses):
        for num in [7,8,9,10]:
            cx, cy = get_pose_target(pose,num)
            _, _, d = get_real_xyz(cx, cy)
        if target == -1 or (d != 0 and d < target_d):
            target = i
            target_d = d
    if target == -1: return None
    return poses[target]
def get_distance(px,py,pz,ax,ay,az,bx,by,bz):
    A,B,C,p1,p2,p3,qx,qy,qz,distance=0,0,0,0,0,0,0,0,0,0
    if ax<=0 or bx<=0 or az==0 or bz==0 or pz==0:
        return 0
    A=int(bx)-int(ax)
    B=int(by)-int(ay)
    C=int(bz)-int(az)
    p1=int(A)*int(px)+int(B)*int(py)+int(C)*int(pz)
    p2=int(A)*int(ax)+int(B)*int(ay)+int(C)*int(az)
    p3=int(A)*int(A)+int(B)*int(B)+int(C)*int(C)
    #print(p1,p2,p3)
    if (p1-p2)!=0 and p3!=0:
        t=(int(p1)-int(p2))/int(p3)
        qx=int(A)*int(t) + int(ax)
        qy=int(B)*int(t) + int(ay)
        qz=int(C)*int(t) + int(az)
        distance = int(pow(((int(px)-int(qx))**2 +(int(py)-int(qy))**2+(int(pz)-int(qz))**2),0.5))
        return int(distance)
    return 0
    
if __name__ == "__main__": 
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    
    _frame = None
    rospy.Subscriber("/cam1/rgb/image_raw", Image, callback_image)

    _depth = None
    rospy.Subscriber("/cam1/depth/image_raw", Image, callback_depth)
    
    _cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    rospy.sleep(1)
    net_pose = HumanPoseEstimation()
    
    dnn_rcnn = Yolov8("yolov8n")
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        if _frame is None: continue
        if _depth is None: continue
        frame = _frame.copy()
        poses = net_pose.forward(frame)
        pose = get_target(poses)
        detections = dnn_rcnn.forward(frame)[0]["det"]
        #[0, tensor(1), tensor(0.9491, grad_fn=<UnbindBackward>), 2, 102, 100, 264]
        #[0, tensor(1), tensor(0.8638, grad_fn=<UnbindBackward>), 118, 121, 194, 202]
        dis=[]
        min1=999999999
        bx, by, bz,ax, ay, az, px,py,pz =0,0,0,0,0,0,0,0,0
        if len(detections) < 2: continue
        for i, detection in enumerate(detections):
            #print(detection)
            x1, y1, x2, y2, score, class_id = map(int, detection)
            score = detection[4]
            if pose is not None:
                #for num in [7,9]: #[7,9] left, [8,10] right
                cx7, cy7 = get_pose_target(pose,7)
                #print(cx7,cy7)
                cx9, cy9 = get_pose_target(pose,9)
                #print(cx9,cy9)
                if cx7==-1 and cx9!=-1:
                    cx5, cy5 = get_pose_target(pose,5)
                    cv2.circle(frame, (cx5,cy5), 5, (0, 255, 0), -1)
                    ax,ay,az = get_real_xyz(cx5, cy5)
                    cv2.circle(frame, (cx9,cy9), 5, (0, 255, 0), -1)
                    bx, by, bz = get_real_xyz(cx9, cy9)
                elif cx7 !=-1 and cx9 ==-1:
                    cx5, cy5 = get_pose_target(pose,5)
                    cv2.circle(frame, (cx5,cy5), 5, (0, 255, 0), -1)
                    bx,by,bz = get_real_xyz(cx5, cy5)
                    cv2.circle(frame, (cx7,cy7), 5, (0, 255, 0), -1)
                    ax, ay, az = get_real_xyz(cx7, cy7)
                elif cx7 ==-1 and cx9 == -1:
                    print("where is your hand")
                    continue
                else:
                    cv2.circle(frame, (cx7,cy7), 5, (0, 255, 0), -1)
                    ax, ay, az = get_real_xyz(cx7, cy7)
                    cv2.circle(frame, (cx9,cy9), 5, (0, 255, 0), -1)
                    bx, by, bz = get_real_xyz(cx9, cy9)
            if class_id==39: #name=="suitcase" or name=="backpack":
                cx1 = (x2 - x1) // 2 + x1
                cy1 = (y2 - y1) // 2 + y1
                cv2.circle(frame, (cx1, cy1), 5, (0, 255, 0), -1)
                px,py,pz=get_real_xyz(cx1, cy1)
                cnt=get_distance(px,py,pz,ax,ay,az,bx,by,bz)
                #print(cnt)
                dis.append(cnt)
                if cnt<=min1 and cnt<=600 and cnt!=0:
                    if str(cnt)!="nan":
                        cv2.putText(frame, str(int(cnt)//10), (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                        min1=cnt
                        time.sleep(1)
                else:
                    if str(cnt)!="nan":
                        cv2.putText(frame, str(int(cnt)//10), (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
            for i,ii in enumerate(dis):
                if(str(i)!="nan" and str(ii)!="nan"):
                    print("num"+str(i),int(ii)//10,end=" ")
            print("")
        cv2.imshow("frame", frame)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break
    
    rospy.loginfo("demo node end!")

