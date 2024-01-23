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
    cx7,cy7,cx9,cy9,cx5,cy5,l,r=0,0,0,0,0,0,0,0
    s1,s2,s3,s4=0,0,0,0
    global ax,ay,az,bx,by,bz,outframe
    #for num in [7,9]: #[7,9] left, [8,10] right
    n1,n2,n3=6,8,10
    cx7, cy7 = get_pose_target(pose,n2)
    
    cx9, cy9 = get_pose_target(pose,n3)
    
    cx5, cy5 = get_pose_target(pose,n1)
    if cx7==-1 and cx9!=-1:
        s1,s2,s3,s4=cx5,cy5,cx9,cy9
        cv2.circle(outframe, (cx5,cy5), 5, (0, 255, 0), -1)
        ax,ay,az = get_real_xyz(depth2,cx5, cy5)
        _,_,l=get_real_xyz(depth2,cx5,cy5)
        cv2.circle(outframe, (cx9,cy9), 5, (0, 255, 0), -1)
        bx, by, bz = get_real_xyz(depth2,cx9, cy9)
        _,_,r=get_real_xyz(depth2,cx9,cy9)
    elif cx7 !=-1 and cx9 ==-1:
        s1,s2,s3,s4=cx5,cy5,cx7,cy7
        cv2.circle(outframe, (cx5,cy5), 5, (0, 255, 0), -1)
        ax, ay, az = get_real_xyz(depth2,cx5, cy5)
        _,_,l=get_real_xyz(depth2,cx5,cy5)
        cv2.circle(outframe, (cx7,cy7), 5, (0, 255, 0), -1)
        bx,by,bz = get_real_xyz(depth2,cx7, cy7)
        _,_,r=get_real_xyz(depth2,cx7,cy7)
    elif cx7 ==-1 and cx9 == -1:
        pass
        #continue
    else:
        s1,s2,s3,s4=cx7,cy7,cx9,cy9
        cv2.circle(outframe, (cx7,cy7), 5, (0, 255, 0), -1)
        ax, ay, az = get_real_xyz(depth2,cx7, cy7)
        _,_,l=get_real_xyz(depth2,cx7,cy7)
        cv2.circle(outframe, (cx9,cy9), 5, (0, 255, 0), -1)
        bx, by, bz = get_real_xyz(depth2,cx9, cy9)
        _,_,r=get_real_xyz(depth2,cx9,cy9)
        
    cv2.putText(outframe, str(int(l)), (s1 + 5, s2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
    cv2.putText(outframe, str(int(r)), (s3 + 5, s4 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
    
    return ax, ay, az, bx, by, bz
def say(a):
    global publisher_speaker
    publisher_speaker.publish(a)

def move(forward_speed: float = 0, turn_speed: float = 0):
    global _cmd_vel
    msg = Twist()
    msg.linear.x = forward_speed
    msg.angular.z = turn_speed
    _cmd_vel.publish(msg)

    
if __name__ == "__main__": 
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    
    frame2 = None
    rospy.Subscriber("/camera/color/image_raw", Image, callback_image2)

    depth2 = None
    rospy.Subscriber("/camera/depth/image_raw", Image, callback_depth2)
    
    
    publisher_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)
    print("load")
    dnn_yolo = Yolov8("yolov8n", device_name="GPU")
    print("yolo")
    say("start the program")
    net_pose = HumanPoseEstimation(device_name="GPU")
    ax,ay,az,bx,by,bz=0,0,0,0,0,0
    b1,b2,b3,b4=0,0,0,0
    _cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    step="dead"
    pre_z, pre_x=0,0
    cur_z, cur_x=0,0
    test=0
    while not rospy.is_shutdown():
        rospy.Rate(10).sleep()
        
        if frame2 is None: 
            print("frame2")
            continue
        if depth2 is None: 
            print("depth2")
            continue
        
        bottle=[]
        detections = dnn_yolo.forward(frame2)[0]["det"]
        al=[]
        ind=0
        for i, detection in enumerate(detections):
            x1, y1, x2, y2, score, class_id = map(int, detection)
            if class_id != 39: continue
            al.append([x1, y1, x2, y2, score, class_id])
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
        if step=="dead":
            outframe=frame2.copy()
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
                ax, ay, az, bx, by, bz = pose_draw()
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
                step="get"
                gg=bb
            print("b1: %d b2: %d b3: %d" % (b1, b2, b3))
        if step=="get":
            if len(bb)!=3: continue
            print(bb)
            h,w,c = outframe.shape
            x1, y1, x2, y2, score, class_id = map(int, bb[mark])
            cx2 = (x2 - x1) // 2 + x1
            cy2 = (y2 - y1) // 2 + y1
            e = w//2-cx2
            v = 0.001 * e
            if v > 0:
                v = min(v, 0.2)
            if v < 0:
                v = max(v, -0.2)
            move(0, v)
            if abs(e) <= 10:
                step="go"
        if step=="go":
            
            cx, cy = w // 2, h // 2
            for i in range(cy + 1, h):
                if depth2[cy][cx] == 0 or 0 < depth2[i][cx] < depth2[cy][cx]:
                    cy = i 
            _,_,d = get_real_xyz(depth2,cx,cy)
            e = d - 400 #number is he last distance
            if abs(e)<=20:
                break
            v = 0.001 * e
            if v > 0:
                v = min(v, 0.2)
            if v < 0:
                v = max(v, -0.2)
            move(v, 0)
        if step=="gettt":
            t=3.0
            time.sleep(3.0)
            publisher_speaker.publish("can you give me the bag")
            joint1, joint2, joint3, joint4 = 0.000, -1.0, 0.3,0.70
            set_joints(joint1, joint2, joint3, joint4, t)
            time.sleep(t)
            open_gripper(t)
            break
            
        cv2.imshow("image", outframe)   
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            break
          
            
        

