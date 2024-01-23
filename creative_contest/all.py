#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import Yolov8,HumanPoseEstimation
from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest
from open_manipulator_msgs.srv import SetKinematicsPose, SetKinematicsPoseRequest
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
    
    cv2.putText(frame2, str(int(l)), (s1 + 5, s2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
    cv2.putText(frame2, str(int(r)), (s3 + 5, s4 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
    return ax, ay, az, bx, by, bz
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

def move(forward_speed: float = 0, turn_speed: float = 0):
    global _cmd_vel
    msg = Twist()
    msg.linear.x = forward_speed
    msg.angular.z = turn_speed
    _cmd_vel.publish(msg)
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
    net_pose = HumanPoseEstimation(device_name="GPU")
    step="fall"
    f_cnt=0
    step2="dead"
    ax,ay,az,bx,by,bz=0,0,0,0,0,0
    b1,b2,b3,b4=0,0,0,0
    pre_z, pre_x=0,0
    cur_z, cur_x=0,0
    test=0
    _cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
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
        if step=="fall":
            print(f_cnt)
            if f_cnt>=5: step="get"
            detections = dnn_yolo.forward(frame2)[0]["det"]
            
            show=frame2
            showd=depth2
            for i, detection in enumerate(detections):
                #time.sleep(0.001)
                fall,ikun=0,0
                x1, y1, x2, y2, score, class_id = map(int, detection)
                score = detection[4]
                if class_id != 0:
                    continue
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                px,py,pz=get_real_xyz(depth2, cx, cy)
                print(pz)
                if pz<=1800:
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
                        _,cy,_,_,dy,_=pose_draw()
                        if abs(cy-dy)<=50: ikun=99
                        
                    #print(x2,x1,y1,y2)
                    w=x2-x1
                    h=y2-y1
                    #print("w: ",w,"h: ", h)
                    w,h=w,h
                    if cy<=160:
                        fall+=1
                        #print("cy")
                    if h<w:
                        fall+=1
                        #print("h<w")
                    if fall>=1 and ikun==99:
                        cv2.rectangle(frame2, (x1, y1), (x2, y2), (255, 255, 0), 5)
                        f_cnt+=1
                        continue
                    else:
                        cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 0,255), 1)
                        continue

        if step=="get":
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
            if step2=="dead":
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
                    step2="turn"
                    gg=bb
                print("b1: %d b2: %d b3: %d" % (b1, b2, b3))
            if step2=="turn":
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
                    step2="go"
            if step2=="go":
                
                cx, cy = w // 2, h // 2
                for i in range(cy + 1, h):
                    if depth2[cy][cx] == 0 or 0 < depth2[i][cx] < depth2[cy][cx]:
                        cy = i 
                _,_,d = get_real_xyz(depth2,cx,cy)
                e = d - 400 #number is he last distance
                if abs(e)<=20:
                    step="grap"
                    step2="brian nigga"
                v = 0.001 * e
                if v > 0:
                    v = min(v, 0.2)
                if v < 0:
                    v = max(v, -0.2)
                move(v, 0)
        if step=="grap":
            t=3.0
            time.sleep(3.0)
            joint1, joint2, joint3, joint4 = 0.000, 1.0, -0.5,-0.6
            set_joints(joint1, joint2, joint3, joint4, t)
            time.sleep(t)
            close_gripper(t)
            break
            step="EEEEEEe"
        if step=="givehim":
            angle_rad = math.atan(60/1)
            angle_deg = math.degrees(angle_rad)
            detections = dnn_yolo.forward(frame2)[0]["det"]
            dc=999999
            mcx,mcy=0,0
            for i, detection in enumerate(detections):
                x1, y1, x2, y2, score, class_id = map(int, detection)
                if class_id != 0: continue
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                _,_,gd=get_real_xyz(depth2,cx,cy)
                if gd<dc:
                    dc=gd
                    mcx,mcy=cx,cy
            rx, ry, rz = get_real_xyz(depth2,mcx, mcy)
            if(rz==0): continue
            angle = np.arctan2(rx, rz)
            print(angle)
            msg.angular.z=-angle
            _cmd_vel.publish(msg)
            step == "put"
        '''
        if step=="put":
            joint1, joint2, joint3, joint4 = 0.000, 0.0, -0.5,1.0
            set_joints(joint1, joint2, joint3, joint4, t)
            time.sleep(t)
            open_gripper(t)
            time.sleep(2)
            joint1, joint2, joint3, joint4 = 0.000, -1.0, 0.3,0.70
            set_joints(joint1, joint2, joint3, joint4, t)
            
            time.sleep(2.5)
            step="back"'''
        if step=="get":
            E=outframe.copy()
        else:
            E=frame2.copy()
        cv2.imshow("image", E)   
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            break
        

