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
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion
import subprocess

def say(a):
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
def callback_imu(msg):
    global _imu
    _imu = msg

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
def turn_to(angle: float, speed: float):
    global _imu
    max_speed = 0.2
    limit_time = 3
    start_time = rospy.get_time()
    while True:
        q = [
            _imu.orientation.x,
            _imu.orientation.z,
            _imu.orientation.y,
            _imu.orientation.w
        ]
        roll, pitch, yaw = euler_from_quaternion(q)
        e = angle - yaw
        print(yaw, e)
        if yaw < 0 and angle > 0:
            cw = np.pi + yaw + np.pi - angle
            aw = -yaw + angle
            if cw < aw:
                e = -cw
        elif yaw > 0 and angle < 0:
            cw = yaw - angle
            aw = np.pi - yaw + np.pi + angle
            if aw < cw:
                e = aw
        if abs(e) < 0.01 or rospy.get_time() - start_time > limit_time:
            break
        move(0.0, max_speed * speed * e)
        rospy.Rate(20).sleep()
    move(0.0, 0.0)


def turn(angle: float):
    global _imu
    q = [
        _imu.orientation.x,
        _imu.orientation.y,
        _imu.orientation.z,
        _imu.orientation.w
    ]
    roll, pitch, yaw = euler_from_quaternion(q)
    target = yaw + angle
    if target > np.pi:
        target = target - np.pi * 2
    elif target < -np.pi:
        target = target + np.pi * 2
    turn_to(target, 0.1)
def close_gripper(t):
    return set_gripper(-0.01, t)
def calc_linear_x(cd,td):
        if cd <= 0: return 0
        e = cd - td
        p = 0.0005
        x = p * e
        if x > 0: x = min(x, 0.5)
        if x < 0: x = max(x, -0.5)
        return x

def calc_angular_z(cx,tx):
    if cx < 0: return 0
    e = tx - cx
    p = 0.0025
    z = p * e
    if z > 0: z = min(z, 0.3)
    if z < 0: z = max(z, -0.3)
    return z
    
def move_to(x, y, z, t):
    service_name = "/goal_task_space_path_position_only"
    rospy.wait_for_service(service_name)
    
    try:
        service = rospy.ServiceProxy(service_name, SetKinematicsPose)
        
        request = SetKinematicsPoseRequest()
        request.end_effector_name = "gripper"
        request.kinematics_pose.pose.position.x = x
        request.kinematics_pose.pose.position.y = y
        request.kinematics_pose.pose.position.z = z
        request.path_time = t
        
        response = service(request)
        return response
    except Exception as e:
        rospy.loginfo("%s" % e)
        return False
#astra
def callback_image1(msg):
    global frame1
    frame1 = CvBridge().imgmsg_to_cv2(msg, "bgr8")

def callback_depth1(msg):
    global depth1
    depth1 = CvBridge().imgmsg_to_cv2(msg, "passthrough")
if __name__ == "__main__": 
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    
    frame2 = None
    rospy.Subscriber("/camera/color/image_raw", Image, callback_image2)

    depth2 = None
    rospy.Subscriber("/camera/depth/image_raw", Image, callback_depth2)
    
    print("load")
    dnn_yolo = Yolov8("yolov8n", device_name="GPU")
    print("yolo")
    net_pose = HumanPoseEstimation(device_name="GPU")
    step="get" #remember
    f_cnt=0
    step2="dead" #remember
    ax,ay,az,bx,by,bz=0,0,0,0,0,0
    b1,b2,b3,b4=0,0,0,0
    pre_z, pre_x=0,0
    cur_z, cur_x=0,0
    test=0
    p_list=[]
    _cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    topic_imu = "/imu/data"
    _imu=None
    rospy.Subscriber(topic_imu, Imu, callback_imu)
    rospy.wait_for_message(topic_imu, Imu)
    say("start the program")
    move_to(0.287,0,0.193,1.0)
    time.sleep(2)
    move_to(0.20,0.0,0.17,1.0)
    time.sleep(2)
    sb=0
    framecnt=0
    bottlecolor=["blue","orange","pink"]
    saidd=0
    get_b=0
    bottlecnt=0
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
        line_img = np.zeros_like(line_frame)
        if step=="get":
            frame2=frame2.copy()
            bottle=[]
            detections = dnn_yolo.forward(frame2)[0]["det"]
            #detections = dnn_yolo.forward(frame)[0]["det"]
            print(detections)
            al=[]
            ind=0
            for i, detection in enumerate(detections):
                x1, y1, x2, y2, score, class_id = map(int, detection)
                score=detection[4]
                if class_id != 39: continue
                if score<0.3: continue
                al.append([x1, y1, x2, y2, score, class_id])
                print(float(score), class_id)
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
                            say("not enught bottle")
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
                if sb == 0:
                    
                    
                    if mark==0: say("右邊那紅色的藥")
                    if mark==1: say("中間那錄色的藥")
                    if mark==2: say("左边那藍色的藥")
                    sb+=1
                    
                if len(bb)!=3: continue
                #print(bb)
                h,w,c = outframe.shape
                x1, y1, x2, y2, score, class_id = map(int, bb[mark])
                if framecnt==0:
                    face_box = [x1, y1, x2, y2]
                    box_roi = outframe[face_box[1]:face_box[3] - 1, face_box[0]:face_box[2] - 1, :]
                    fh,fw=abs(x1-x2),abs(y1-y2)
                    box_roi=cv2.resize(box_roi, (fh*10,fw*10), interpolation=cv2.INTER_AREA)
                    cv2.imshow("bottle", box_roi)  
                    get_b=mark
                    framecnt+=1
                cx2 = (x2 - x1) // 2 + x1
                cy2 = (y2 - y1) // 2 + y1
                e = w//2-cx2
                v = 0.0015 * e
                if v > 0:
                    v = min(v, 0.2)
                if v < 0:
                    v = max(v, -0.2)
                move(0, v)
                if abs(e) <= 3:
                    step2="go"
                    
            if step2=="go":
                
                cx, cy = w // 2, h // 2
                for i in range(cy + 1, h):
                    if depth2[cy][cx] == 0 or 0 < depth2[i][cx] < depth2[cy][cx]:
                        cy = i 
                _,_,d = get_real_xyz(depth2,cx,cy)
                e = d - 400 #number is he last distance
                if abs(e)<=15:
                    say("turn_again")
                    step2="turn_again"
                    
                    
                v = 0.001 * e
                if v > 0:
                    v = min(v, 0.25)
                if v < 0:
                    v = max(v, -0.25)
                move(v, 0)
                #print(d, v)
                #cv2.imshow("turn", outframe) 
            if step2=="turn_again":
                if len(bb)<1: continue
                min11=999999
                ucx,ucy=0,0
                for num in bb:
                    x1, y1, x2, y2, score, class_id = map(int, num)
                    cx2 = (x2 - x1) // 2 + x1
                    cy2 = (y2 - y1) // 2 + y1
                    _,_,d = get_real_xyz(depth2,cx2,cy2)
                    if abs(w//2-cx2) < min11:
                        min11=abs(w//2-cx2)
                        ucx = cx2
                        
                h,w,c = outframe.shape
                e = w//2 - ucx
                v = 0.001 * e
                if v > 0:
                    v = min(v, 0.2)
                if v < 0:
                    v = max(v, -0.2)
                move(0, v)
                if abs(e) <= 4:
                    step="grap"
                    get_b=mark
                    step2="none"
                    move(0,0.05)
        if step=="grap":
            say("我現在會拿給你")
            t=3.0
            open_gripper(t)
            time.sleep(3.0)
            h,w,c = frame2.shape
            cx, cy = w // 2, h // 2
            for i in range(cy + 1, h):
                if depth2[cy][cx] == 0 or 0 < depth2[i][cx] < depth2[cy][cx]:
                    cy = i 
            _,_,d = get_real_xyz(depth2,cx,cy)
            x,y,z=d/1000.0, 0.019, 0.06 #z 水平 y 上下 x 前後
            for i in range(1000): move(0.2,0)
            if x >= 0.25:
                cntm=int((x-0.25)*1000//0.2)
                for i in range(cntm): move(0.2,0)
            
            move_to(0.25,0.0,0.175,3.0)
            time.sleep(2)
            print(x)
            time.sleep(t)
            close_gripper(t)
            time.sleep(2)
            
            saidd="我拿了藍色的藥"
            say(saidd)
            t=3.0
            joint1, joint2, joint3, joint4 = 0.000, -1.0,0.357,0.703
            set_joints(joint1, joint2, joint3, joint4, t)
            time.sleep(t)
            break
            step="givehim"
            #break
        if step=="givehim":
            turn(-30)
            step="findg"
        if step=="findg":
            cx,cy=0,0
            rcx,rcy=0,0
            max_=-1
            frameg = frame2.copy()
            last_distance=999999
            detections = dnn_yolo.forward(frameg)[0]["det"]
            yn="no"
            h, w = frame2.shape[:2]
            for i, detection in enumerate(detections):
                #print(detection)
                x1, y1, x2, y2, score, class_id = map(int, detection)
                score = detection[4]
                #print(x1, y1, x2, y2, score, class_id)
                cx = (x2 - x1) // 2 + x1
                cy = (y2 - y1) // 2 + y1
                _,_,hg=get_real_xyz(depth2,cx,cy)
                if score > 0.5 and class_id == 0 and hg<=2500: 
                    step2="turn"
                    #dnn_yolo.draw_bounding_box(detection, frame)
                    cv2.rectangle(frameg, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.circle(frameg, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frameg, "person", (x1+5, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    hh=x2-x1
                    ll=y2-y1
                    if hg<last_distance and hg<1000:
                        yn="yes"
                        max_=ll*hh
                        rcx,rcy=cx,cy
                        last_distance=hg
                        
            if rcx==0: continue
            _, _, d = get_real_xyz(depth2, rcx, rcy)
            if d<=350 and d>=0: 
                say("here you are")
                step="put"
            print("people: depth: ",d)
            cur_x = calc_linear_x(d, 300)
            cur_z = calc_angular_z(cx, 320)

            dx = cur_x - pre_x
            if dx > 0: dx = min(dx, 0.03)
            if dx < 0: dx = max(dx, -0.03)
            
            dz = cur_z - pre_z
            if dz > 0: dz = min(dz, 0.2)
            if dz < 0: dz = max(dz, -0.2)

            cur_x = pre_x + dx
            cur_z = pre_z + dz
            
            if yn=="no":
                cur_x,cur_z=0,-0.2
                print("no people")
                
            
            pre_x = cur_x 
            pre_z = cur_z 
            move(cur_x,cur_z)
        if step=="put":
            t=3.0
            time.sleep(3.0)
            joint1, joint2, joint3, joint4 = 0.000, 1.0, -0.5,-0.6
            set_joints(joint1, joint2, joint3, joint4, t)
            time.sleep(t)
            open_gripper(t)
            time.sleep(2)
            joint1, joint2, joint3, joint4 = 0.000, -1.0, 0.3,0.70
            set_joints(joint1, joint2, joint3, joint4, t)
            
            time.sleep(2.5)
            break
        
        if step=="get" and step2=="dead":
            E=outframe.copy()
        else:
            E=frame2.copy()
        
        cv2.imshow("image", E)   
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            break
        

