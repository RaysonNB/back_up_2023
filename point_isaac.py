#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import Yolov8,HumanPoseEstimation
import numpy as np
from geometry_msgs.msg import Twist
from pcms.pytorch_models import *
from pcms.openvino_yolov8 import *
import math

#https://github.com/supercatex/code/blob/master/ros/demo3.py
def callback_image(msg):
    global _frame
    _frame = CvBridge().imgmsg_to_cv2(msg, "bgr8")

def callback_depth(msg):
    global _depth
    _depth = CvBridge().imgmsg_to_cv2(msg, "passthrough")
    
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
    
def get_pose_target(pose,num):
    p = []
    for i in [num]:
        if pose[i][2] > 0:
            p.append(pose[i])
    
    if len(p) == 0: return -1, -1
    return int(p[0][0]),int(p[0][1])
    
def get_distance(px,py,pz,ax,ay,az,bx,by,bz):
    A,B,C,p1,p2,p3,qx,qy,qz,distance=0,0,0,0,0,0,0,0,0,0
    A=int(bx)-int(ax)
    B=int(by)-int(ay)
    C=int(bz)-int(az)
    p1=int(A)*int(px)+int(B)*int(py)+int(C)*int(pz)
    p2=int(A)*int(ax)+int(B)*int(ay)+int(C)*int(az)
    p3=int(A)*int(A)+int(B)*int(B)+int(C)*int(C)
    #print("1",p1,p2,p3)
    if (p1-p2)!=0 and p3!=0:
        t=(int(p1)-int(p2))/int(p3)
        qx=int(A)*int(t) + int(ax)
        qy=int(B)*int(t) + int(ay)
        qz=int(C)*int(t) + int(az)
        return int(int(pow(((int(qx)-int(px))**2 +(int(qy)-int(py))**2+(int(qz)-int(pz))**2),0.5)))
    return 0
    
def pose_draw():
    cx7,cy7,cx9,cy9,cx5,cy5=0,0,0,0,0,0
    global ax,ay,az,bx,by,bz
    #for num in [7,9]: #[7,9] left, [8,10] right
    n1,n2,n3=6,8,10
    cx7, cy7 = get_pose_target(pose,n2)
    
    cx9, cy9 = get_pose_target(pose,n3)
    
    cx5, cy5 = get_pose_target(pose,n1)
    if cx7==-1 and cx9!=-1:
        cv2.circle(frame, (cx5,cy5), 5, (0, 255, 0), -1)
        ax,ay,az = get_real_xyz(cx5, cy5)
        
        cv2.circle(frame, (cx9,cy9), 5, (0, 255, 0), -1)
        bx, by, bz = get_real_xyz(cx9, cy9)
    elif cx7 !=-1 and cx9 ==-1:

        cv2.circle(frame, (cx5,cy5), 5, (0, 255, 0), -1)
        ax, ay, az = get_real_xyz(cx5, cy5)
        
        cv2.circle(frame, (cx7,cy7), 5, (0, 255, 0), -1)
        bx,by,bz = get_real_xyz(cx7, cy7)
    elif cx7 ==-1 and cx9 == -1:
        pass
        #continue
    else:
        cv2.circle(frame, (cx7,cy7), 5, (0, 255, 0), -1)
        ax, ay, az = get_real_xyz(cx7, cy7)
        
        cv2.circle(frame, (cx9,cy9), 5, (0, 255, 0), -1)
        bx, by, bz = get_real_xyz(cx9, cy9)

if __name__ == "__main__": 
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    
    _frame = None
    rospy.Subscriber("/camera/color/image_raw", Image, callback_image)

    _depth = None
    rospy.Subscriber("/camera/depth/image_raw", Image, callback_depth)
    
    print("load")
    dnn_yolo = Yolov8("yolov8n", device_name="GPU")
    print("load2")
    net_pose = HumanPoseEstimation(device_name="GPU")
    print("yolo")

    ax,ay,az,bx,by,bz=0,0,0,0,0,0
    rate = rospy.Rate(30)
    yolo_cnt = 0
    while not rospy.is_shutdown():
        rate.sleep()
        
        if _frame is None: continue
        if _depth is None: continue
        
        cnt = 1
        frame = _frame#.copy()
        depth=_depth#.copy()
        
        min1=99999999
        
        TTT=-1
        
        bottle=[]
        if yolo_cnt % 30 == 0:
            detections = dnn_yolo.forward(frame)[0]["det"]
        yolo_cnt += 2
        pose=None
        for i, detection in enumerate(detections):
            #print(detection)
            x1, y1, x2, y2, score, class_id = map(int, detection)
            score = detection[4]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if class_id ==0 and int(depth[cy][cx])<=1500:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 2), 2)

        poses = net_pose.forward(frame)
        #if len(detections) < 2: continue
        for i, detection in enumerate(detections):
            #print(detection)
            x1, y1, x2, y2, score, class_id = map(int, detection)
            score = detection[4]
            if class_id == 0:
                continue
            if class_id !=39: continue
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            px,py,pz=get_real_xyz(cx, cy)
            cnt=get_distance(px,py,pz,ax,ay,az,bx,by,bz)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
            #cv2.putText(frame, str(int(cnt)//10), (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
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
        TTT=0
        E=0
        s_c=[]
        ggg=0
        flag=None
        if pose is not None:
            pose_draw()
            for i, detection in enumerate(detections):
                #print(detection)
                x1, y1, x2, y2, score, class_id = map(int, detection)
                score = detection[4]
                #print(id)
                if(class_id == 39):
                    ggg=1
                    bottle.append(detection)
                    E+=1
                    cx1 = (x2 - x1) // 2 + x1
                    cy1 = (y2 - y1) // 2 + y1

                    px,py,pz=get_real_xyz(cx1, cy1)
                    cnt=get_distance(px,py,pz,ax,ay,az,bx,by,bz)
                    cv2.putText(frame, str(int(cnt)), (x1+5, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 3)
                    cnt=int(cnt)
                    s_c.append(cnt)
                    #print(cnt)

            if ggg==0:
                s_c=[9999]
            TTT=min(s_c)
            E=s_c.index(TTT)
            for i, detection in enumerate(bottle):
                #print("1")
                x1, y1, x2, y2, score, class_id = map(int, detection)
                if(class_id == 39):
                    if i == E and E!=9999:
                        #cv2.putText(frame, str(int(cnt)//10), (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                        break
        
        cv2.imshow("frame", frame)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break
    
    rospy.loginfo("demo node end!")

