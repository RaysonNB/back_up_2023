#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import Yolov8,HumanPoseEstimation
import numpy as np
from pcms.pytorch_models import *
from pcms.openvino_yolov8 import *
import math
import time
from mr_voice.msg import Voice
from std_msgs.msg import String
import subprocess
import random

mleft = []
mright = []

def draw(event,x,y,flags,param):
    if(event == cv2.EVENT_LBUTTONDBLCLK and mleft == []):
        mleft.append(x)
        mleft.append(y)
    elif(event == cv2.EVENT_LBUTTONDBLCLK and mleft != [] and mright == []):
        mright.append(x)
        mright.append(y)

def dfs(x, y, statu):
    global depth_copy, depth_list, cnt
    if x < 1 or y < 1 or x > len(depth_copy[0]) - 2 or y > len(depth_copy) - 2:
        return
    if depth_copy[y][x] != 0:
        return
    depth_copy[y][x] = statu
    cnt += 1
    if x < 2:
        dfs(x + 1, y, statu)
        return
    if y < 2:
        dfs(x, y + 1, statu)
        return

    bx = False
    by = False
    if abs(abs(depth_list[y + 1][x] - depth_list[y][x]) - abs(depth_list[y][x] - depth_list[y - 1][x])) < 3:
        by = True
        dfs(x, y - 1, statu)
        dfs(x, y + 1, statu)
    if abs(abs(depth_list[y][x + 1] - depth_list[y][x]) - abs(depth_list[y][x] - depth_list[y][x - 1])) < 3:
        bx = True
        dfs(x + 1, y, statu)
        dfs(x - 1, y, statu)
    if not bx and not by:
        return
    return

def change_zero():
    for e in range(1, len(depth_list) - 1, 1):
        error = []
        for f in range(1, len(depth_list[e]) - 1, 1):
            if depth_list[e][f] == 0:
                if depth_list[e - 1][f] or depth_list[e - 1][f - 1] or depth_list[e - 1][f + 1] or depth_list[e + 1][f] or depth_list[e + 1][f - 1] or depth_list[e + 1][f + 1] or depth_list[e][f - 1] or depth_list[e][f + 1]:
                    for i in range(-1, 2, 1):
                        for j in range(-1, 2, 1):
                            if depth_list[e + i][f + j] != 0:
                                error.append(depth_list[e + i][f + j])
                    depth_list[e][f] = sum(error) // len(error)
def say_cn(a):
    text = str(a)
    process = subprocess.Popen(['espeak-ng', '-v', 'yue', text])
    process.wait()
#gemini2
def callback_image2(msg):
    global frame2
    frame2 = CvBridge().imgmsg_to_cv2(msg, "bgr8")

def callback_depth2(msg):
    global depth2
    depth2 = CvBridge().imgmsg_to_cv2(msg, "passthrough")
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

def pose_draw(show):
    cx7,cy7,cx9,cy9,cx5,cy5,l,r=0,0,0,0,0,0,0,0
    s1,s2,s3,s4=0,0,0,0
    global ax,ay,az,bx,by,bz
    #for num in [7,9]: #[7,9] left, [8,10] right
    n1,n2,n3=6,8,10
    cx7, cy7 = get_pose_target(pose,n2)
    
    cx9, cy9 = get_pose_target(pose,n3)
    
    cx5, cy5 = get_pose_target(pose,n1)
    if cx7==-1 and cx9!=-1:
        s1,s2,s3,s4=cx5,cy5,cx9,cy9
        cv2.circle(show, (cx5,cy5), 5, (0, 255, 0), -1)
        ax,ay,az = get_real_xyz(depth2,cx5, cy5)
        _,_,l=get_real_xyz(depth2,cx5,cy5)
        cv2.circle(show, (cx9,cy9), 5, (0, 255, 0), -1)
        bx, by, bz = get_real_xyz(depth2,cx9, cy9)
        _,_,r=get_real_xyz(depth2,cx9,cy9)
    elif cx7 !=-1 and cx9 ==-1:
        s1,s2,s3,s4=cx5,cy5,cx7,cy7
        cv2.circle(show, (cx5,cy5), 5, (0, 255, 0), -1)
        ax, ay, az = get_real_xyz(depth2,cx5, cy5)
        _,_,l=get_real_xyz(depth2,cx5,cy5)
        cv2.circle(show, (cx7,cy7), 5, (0, 255, 0), -1)
        bx,by,bz = get_real_xyz(depth2,cx7, cy7)
        _,_,r=get_real_xyz(depth2,cx7,cy7)
    elif cx7 ==-1 and cx9 == -1:
        pass
        #continue
    else:
        s1,s2,s3,s4=cx7,cy7,cx9,cy9
        cv2.circle(show, (cx7,cy7), 5, (0, 255, 0), -1)
        ax, ay, az = get_real_xyz(depth2,cx7, cy7)
        _,_,l=get_real_xyz(depth2,cx7,cy7)
        cv2.circle(show, (cx9,cy9), 5, (0, 255, 0), -1)
        bx, by, bz = get_real_xyz(depth2,cx9, cy9)
        _,_,r=get_real_xyz(depth2,cx9,cy9)
    
    cv2.putText(show, str(int(l)), (s1 + 5, s2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
    cv2.putText(show, str(int(r)), (s3 + 5, s4 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
    return ax, ay, az, bx, by, bz
def get_pose_target(pose,num):
    p = []
    for i in [num]:
        if pose[i][2] > 0:
            p.append(pose[i])
    
    if len(p) == 0: return -1, -1
    return int(p[0][0]),int(p[0][1])
def callback_voice(msg):
    global s
    s = msg.text
def say(a):
    global publisher_speaker
    publisher_speaker.publish(a)
def add_edge():
    for e in range(len(depth_list)):
        depth_list[e].insert(0, depth_list[e][0])  # 最左
        depth_list[e].insert(-1, depth_list[e][-1])  # 最右
    depth_list.insert(0, depth_list[0])
    depth_list.insert(-1, depth_list[-1])
depth_copy = None
depth_list = []
xylist = []
color = {}
biggest_max = []

if __name__ == "__main__": 
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    
    frame2 = None
    rospy.Subscriber("/camera/color/image_raw", Image, callback_image2)

    depth2 = None
    rospy.Subscriber("/camera/depth/image_raw", Image, callback_depth2)
    
    dnn_yolo = Yolov8("yolov8n", device_name="GPU")
    
    print("yolo")
    net_pose = HumanPoseEstimation(device_name="GPU")
    print("pose")
    
    rospy.Subscriber("/voice/text", Voice, callback_voice)
    publisher_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)
    
    print("speaker")
    step="get" #remember
    f_cnt=0
    step2="dead" #remember
    ax,ay,az,bx,by,bz=0,0,0,0,0,0
    b1,b2,b3,b4=0,0,0,0
    pre_z, pre_x=0,0
    cur_z, cur_x=0,0
    test=0
    p_list=[]
    sb=0
    framecnt=0
    bottlecolor=["blue","orange","pink"]
    saidd=0
    get_b=0
    bottlecnt=0
    say_cn("開始！")
    line_destory_cnt=0
    while not rospy.is_shutdown():
        rospy.Rate(10).sleep()
        
        if frame2 is None: 
            print("frame2")
            continue
        if depth2 is None: 
            print("depth2")
            continue
        
        line_frame=frame2.copy()
        frame2=frame2.copy()
        bottle=[]
        detections = dnn_yolo.forward(frame2)[0]["det"]
        #detections = dnn_yolo.forward(frame)[0]["det"]
        #print(detections)
        al=[]
        ind=0
        for i, detection in enumerate(detections):
            x1, y1, x2, y2, score, class_id = map(int, detection)
            score=detection[4]
            if class_id != 39: continue
            if score<0.3: continue
            al.append([x1, y1, x2, y2, score, class_id])
            #print(float(score), class_id)
            cv2.putText(frame2, str(class_id), (x1+5, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        bb=sorted(al, key=(lambda x:x[0]))
        #print(bb)
        for i in bb:
            #print(i)
            x1, y1, x2, y2, _, _ = i
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 255), 4)
            cv2.putText(frame2, str(int(ind)), (cx,cy+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
            ind+=1
            px,py,pz=get_real_xyz(depth2,cx, cy)
            cnt=get_distance(px,py,pz,ax,ay,az,bx,by,bz)
            cv2.circle(frame2, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(frame2, str(int(pz)), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)  
        if step=="get":        
            outframe=frame2.copy()
            if step2=="dead":
                
                t_pose=None
                points=[]
                poses = net_pose.forward(outframe)
                
                for i, pose in enumerate(poses):
                    point = []
                    for j, (x,y,preds) in enumerate(pose): #x: ipex 坐標 y: ipex 坐標 preds: 准度
                        if preds <= 0: continue
                        x,y = map(int,[x,y])
                        _,_,td=get_real_xyz(depth2,x, y)
                        if td>=2000: continue
                        if j in [8,10]:
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
                    ax, ay, az, bx, by, bz = pose_draw(outframe)
                    if len(bb) <3:
                        if bottlecnt>=3:
                            print("not enught bottle")
                            bottlecnt+=1
                        continue
                    for i, detection in enumerate(bb):
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
                            
                            
                            px,py,pz=get_real_xyz(depth2, cx1, cy1)
                            cnt=get_distance(px,py,pz,ax,ay,az,bx,by,bz)
                            
                            cnt=int(cnt)
                            if cnt!=0 and cnt<=600: cnt=int(cnt)
                            else: cnt=9999
                            s_c.append(cnt)
                            s_d.append(pz)
                            
                if ggg==0: s_c=[9999]
                TTT=min(s_c)
                E=s_c.index(TTT)
                for i, detection in enumerate(bottle):
                    #print("1")
                    x1, y1, x2, y2, score, class_id = map(int, detection)
                    if(class_id == 39):
                        if i == E and E!=9999 and TTT <=700:
                            cx1 = (x2 - x1) // 2 + x1
                            cy1 = (y2 - y1) // 2 + y1
                            print("hello")
                            cv2.putText(outframe, str(int(TTT)//10), (x1 + 5, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (0, 0, 255), 2)
                            cv2.rectangle(outframe, (x1, y1), (x2, y2), (0, 0, 255), 5)
                            if i==0: b1+=1
                            if i==1: b2+=1
                            if i==2: b3+=1
                            
                            break
                                    
                        else:
                            v=s_c[i]
                            cv2.putText(outframe, str(int(v)), (x1+5, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if b1==max(b1,b2,b3): mark=0
                if b2==max(b1,b2,b3): mark=1
                if b3==max(b1,b2,b3): mark=2
                if b1 >=10 or b2>=10 or b3>=10: 
                    step2="turn"
                    gg=bb
                #print("b1: %d b2: %d b3: %d" % (b1, b2, b3))
            if step2=="turn":
                if mark==0: say_cn("the left tea, which is the blue one")
                if mark==1: say_cn("the middle tea, which is the orange one")
                if mark==2: say_cn("the right tea, which is the pink one")
                b1,b2,b3=0,0,0
                step2="dead"
                step="get"
                
        if step=="get" and step2=="dead":
            E=outframe.copy()
        else:
            E=frame2.copy()
        
        cv2.imshow("image", E)   
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            break
        

