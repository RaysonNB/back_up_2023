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
import time
from mr_voice.msg import Voice
from std_msgs.msg import String
#gemini2
def callback_image2(msg):
    global frame2
    frame2 = CvBridge().imgmsg_to_cv2(msg, "bgr8")

def callback_depth2(msg):
    global depth2
    depth2 = CvBridge().imgmsg_to_cv2(msg, "passthrough")
#astra
def callback_image1(msg):
    global frame1
    frame1 = CvBridge().imgmsg_to_cv2(msg, "bgr8")

def callback_depth1(msg):
    global depth1
    depth1 = CvBridge().imgmsg_to_cv2(msg, "passthrough")
    
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

def pose_draw():
    cx7,cy7,cx9,cy9,cx5,cy5,l,r=0,0,0,0,0,0,0,0
    s1,s2,s3,s4=0,0,0,0
    global ax,ay,az,bx,by,bz,frame2
    #for num in [7,9]: #[7,9] left, [8,10] right
    n1,n2,n3=6,8,10
    cx7, cy7 = get_pose_target(pose,n2)
    
    cx9, cy9 = get_pose_target(pose,n3)
    
    cx5, cy5 = get_pose_target(pose,n1)
    if cx7==-1 and cx9!=-1:
        s1,s2,s3,s4=cx5,cy5,cx9,cy9
        cv2.circle(frame2, (cx5,cy5), 5, (0, 255, 0), -1)
        ax,ay,az = get_real_xyz(depth2,cx5, cy5)
        _,_,l=get_real_xyz(depth2,cx5,cy5)
        cv2.circle(frame2, (cx9,cy9), 5, (0, 255, 0), -1)
        bx, by, bz = get_real_xyz(depth2,cx9, cy9)
        _,_,r=get_real_xyz(depth2,cx9,cy9)
    elif cx7 !=-1 and cx9 ==-1:
        s1,s2,s3,s4=cx5,cy5,cx7,cy7
        cv2.circle(frame2, (cx5,cy5), 5, (0, 255, 0), -1)
        ax, ay, az = get_real_xyz(depth2,cx5, cy5)
        _,_,l=get_real_xyz(depth2,cx5,cy5)
        cv2.circle(frame2, (cx7,cy7), 5, (0, 255, 0), -1)
        bx,by,bz = get_real_xyz(depth2,cx7, cy7)
        _,_,r=get_real_xyz(depth2,cx7,cy7)
    elif cx7 ==-1 and cx9 == -1:
        pass
        #continue
    else:
        s1,s2,s3,s4=cx7,cy7,cx9,cy9
        cv2.circle(frame2, (cx7,cy7), 5, (0, 255, 0), -1)
        ax, ay, az = get_real_xyz(depth2,cx7, cy7)
        _,_,l=get_real_xyz(depth2,cx7,cy7)
        cv2.circle(frame2, (cx9,cy9), 5, (0, 255, 0), -1)
        bx, by, bz = get_real_xyz(depth2,cx9, cy9)
        _,_,r=get_real_xyz(depth2,cx9,cy9)
    
    return ay,by
    #cv2.putText(frame, str(int(l)), (s1 + 5, s2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
    #cv2.putText(frame, str(int(r)), (s3 + 5, s4 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
def get_pose_target(pose,num):
    p = []
    for i in [num]:
        if pose[i][2] > 0:
            p.append(pose[i])
    
    if len(p) == 0: return -1, -1
    return int(p[0][0]),int(p[0][1])
def say(a):
    global publisher_speaker
    publisher_speaker.publish(a)
if __name__ == "__main__": 
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    
    frame2 = None
    rospy.Subscriber("/camera/color/image_raw", Image, callback_image2)

    depth2 = None
    rospy.Subscriber("/camera/depth/image_raw", Image, callback_depth2)
    
    frame1 = None
    rospy.Subscriber("/cam2/rgb/image_raw", Image, callback_image1)

    depth1= None
    rospy.Subscriber("/cam2/depth/image_raw", Image, callback_depth1)
    
    publisher_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)
    print("load")
    dnn_yolo = Yolov8("yolov8n", device_name="GPU")
    print("yolo")
    say("start the program")
    ffcnt=0
    net_pose = HumanPoseEstimation(device_name="GPU")
    while not rospy.is_shutdown():
        rospy.Rate(10).sleep()
        
        if frame1 is None: 
            print("frame1")
            continue
        if frame2 is None: 
            print("frame2")
            continue
        if depth1 is None: 
            print("depth1")
            continue
        if depth2 is None: 
            print("depth2")
            continue
        
        detections = dnn_yolo.forward(frame2)[0]["det"]
        people_list=[]
        show=frame2
        showd=depth2
        if ffcnt>=2: break
        for i, detection in enumerate(detections):
            #time.sleep(0.001)
            fall,ikun=0,0
            x1, y1, x2, y2, score, class_id = map(int, detection)
            score = detection[4]
            people_list.append([x1,y1,x2,y2,score,class_id]) #test
            if class_id != 0:
                continue
            #people_list.append(detection) #test
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            px,py,pz=get_real_xyz(depth2, cx, cy)
            if pz<=2000:
                pose=None
                poses = net_pose.forward(frame2)
                t_pose=None
                points=[]
                for i, pose in enumerate(poses):
                    point = []
                    for j, (x,y,preds) in enumerate(pose): #x: ipex 坐標 y: ipex 坐標 preds: 准度
                        if preds <= 0: continue
                        x,y = map(int,[x,y])
                        _,_,td=get_real_xyz(depth2,x, y)
                        if td>=2000: continue
                        if j in [6,8]:
                            point.append(j)
                    if len(point) == 2:
                        t_pose = poses[i]
                        break
                    #print(point)
                TTT=0
                E=0
                s_c=[]
                
                s_d=[]
                ggg=0
                flag=None
                if t_pose is not None:
                    cy,dy=pose_draw()
                    if abs(cy-dy)<=50: ikun=99
                    
                #print(x2,x1,y1,y2)
                w=abs(x2)-abs(x1)
                h=abs(y1)-abs(y2)
                #print("w: ",w,"h: ", h)
                w,h=abs(w),abs(h)
                if cy<=160:
                    fall+=1
                    #print("cy")
                if h<w:
                    fall+=1
                    #print("h<w")
                if fall>=1 and ikun==99:
                    cv2.rectangle(frame2, (x1, y1), (x2, y2), (255, 255, 0), 5)
                    ffcnt+=1
                    continue
                else:
                    cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 0,255), 1)
                    continue
        
        r=sorted(people_list, key=(lambda x:x[0]))
        #print(r)
              #cv2.putText(frame, str(int(pz)), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 3)
            #print(fall)
        cv2.imshow("image", frame2)   
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            break
        
