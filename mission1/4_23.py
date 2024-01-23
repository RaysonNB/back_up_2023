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
    
    
def calc_angular_z(cx,tx):
    if cx < 0: return 0
    e = tx - cx
    p = 0.0025
    z = p * e
    if z > 0: z = min(z, 0.3)
    if z < 0: z = max(z, -0.3)
    return z

def calc_linear_x(cd,td):
    if cd <= 0: return 0
    e = cd - td
    p = 0.0005
    x = p * e
    if x > 0: x = min(x, 0.5)
    if x < 0: x = max(x, -0.5)
    return x
def callback_voice(msg):
    global s
    s = msg.text

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

def open_gripper(t):
    return set_gripper(0.01, t)

def close_gripper(t):
    return set_gripper(-0.01, t)
def get_real_xyz(x, y):
    global _depth1
    a = 49.5 * np.pi / 180
    b = 60.0 * np.pi / 180
    d = _depth1[y][x]
    h, w = _depth1.shape[:2]
    if d == 0:
        for k in range(1, 15, 1):
            if d == 0 and y - k >= 0:
                for j in range(x - k, x + k, 1):
                    if not (0 <= j < w):
                        continue
                    d = _depth1[y - k][j]
                    if d > 0:
                        break
            if d == 0 and x + k < w:
                for i in range(y - k, y + k, 1):
                    if not (0 <= i < h):
                        continue
                    d = _depth1[i][x + k]
                    if d > 0:
                        break
            if d == 0 and y + k < h:
                for j in range(x + k, x - k, -1):
                    if not (0 <= j < w):
                        continue
                    d = _depth1[y + k][j]
                    if d > 0:
                        break
            if d == 0 and x - k >= 0:
                for i in range(y + k, y - k, -1):
                    if not (0 <= i < h):
                        continue
                    d = _depth1[i][x - k]
                    if d > 0:
                        break
            if d > 0:
                break

    x = int(x) - int(w // 2)
    y = int(y) - int(h // 2)
    real_y = round(y * 2 * d * np.tan(a / 2) / h)
    real_x = round(x * 2 * d * np.tan(b / 2) / w)
    return real_x, real_y, d


def get_pose_target(pose, num):
    p = []
    for i in [num]:
        if pose[i][2] > 0:
            p.append(pose[i])

    if len(p) == 0:
        return -1, -1
    return int(p[0][0]), int(p[0][1])


def get_distance(px, py, pz, ax, ay, az, bx, by, bz):
    A, B, C, p1, p2, p3, qx, qy, qz, distance = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    if ax <= 0 or bx <= 0 or az == 0 or bz == 0 or pz == 0:
        return 0
    A = int(bx)-int(ax)
    B = int(by)-int(ay)
    C = int(bz)-int(az)
    p1 = int(A)*int(px)+int(B)*int(py)+int(C)*int(pz)
    p2 = int(A)*int(ax)+int(B)*int(ay)+int(C)*int(az)
    p3 = int(A)*int(A)+int(B)*int(B)+int(C)*int(C)
    #print(p1,p2,p3)
    if (p1-p2) != 0 and p3 != 0:
        t = (int(p1)-int(p2))/int(p3)
        #print(p1,p2,p3)
        qx = int(A)*int(t) + int(ax)
        qy = int(B)*int(t) + int(ay)
        qz = int(C)*int(t) + int(az)
        distance = int(
            pow(((int(px)-int(qx))**2 + (int(py)-int(qy))**2+(int(pz)-int(qz))**2), 0.5))
        
        return t
    return 0



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
def callback_image(msg):
    global _frame
    _frame = CvBridge().imgmsg_to_cv2(msg, "bgr8")


def pose_draw(f):
    cx7, cy7, cx9, cy9, cx5, cy5 = 0, 0, 0, 0, 0, 0
    global ax, ay, az, bx, by, bz
    #for num in [7,9]: #[7,9] left, [8,10] right
    n1, n2, n3 = 6, 8, 10
    #print(pose)
    cx7, cy7 = get_pose_target(pose, n2)

    cx9, cy9 = get_pose_target(pose, n3)

    cx5, cy5 = get_pose_target(pose, n1)
    
    show=f.copy()
    if cx7 == -1 and cx9 != -1:
        cv2.circle(show, (cx5, cy5), 5, (0, 255, 0), -1)
        ax, ay, az = get_real_xyz(cx5, cy5)

        cv2.circle(show, (cx9, cy9), 5, (0, 255, 0), -1)
        bx, by, bz = get_real_xyz(cx9, cy9)
    elif cx7 != -1 and cx9 == -1:

        cv2.circle(show, (cx5, cy5), 5, (0, 255, 0), -1)
        ax, ay, az = get_real_xyz(cx5, cy5)

        cv2.circle(show, (cx7, cy7), 5, (0, 255, 0), -1)
        bx, by, bz = get_real_xyz(cx7, cy7)
    elif cx7 == -1 and cx9 == -1:
        pass
        #print("no")
        #continue
    else:
        cv2.circle(show, (cx7, cy7), 5, (0, 255, 0), -1)
        ax, ay, az = get_real_xyz(cx7, cy7)

        cv2.circle(show, (cx9, cy9), 5, (0, 255, 0), -1)
        bx, by, bz = get_real_xyz(cx9, cy9)
    return show
def say(a):
    global publisher_speaker
    publisher_speaker.publish(a)


def callback_depth(msg):
    global _depth
    _depth = CvBridge().imgmsg_to_cv2(msg, "passthrough")


def callback_image1(msg):
    global _image1
    _image1 = CvBridge().imgmsg_to_cv2(msg, "bgr8")

def callback_depth1(msg):
    global _depth1
    _depth1 = CvBridge().imgmsg_to_cv2(msg, "passthrough")

def move1(cx, cy, msg):
    global rgb_image, pre_z, pre_x,_depth,f
    h, w, c = f.shape
    e = w // 2 - cx
    d = _depth
    if abs(e) > 30:
        rospy.loginfo("turn")
        v = 0.0015 * e
        cur_z = v
        dz = cur_z - pre_z
        if dz > 0:
            dz = min(dz, 0.1)
        if dz < 0:
            dz = max(dz, -0.1)
        msg.angular.z = pre_z + dz
        print(e)
    else:
        rospy.loginfo("run")
        d = _depth[cy][cx]
        rospy.loginfo("d: %d" % d)

        if d > 0 and d < 3000:
            v1 = 0.0001 * d
            msg.linear.x = v1
            cur_x = v1

            dx = cur_x - pre_x
            if dx > 0:
                dx = min(dx, 0.05)
            if dx < 0:
                dx = max(dx, -0.05)
            msg.linear.x = pre_x + dx
        else:
            msg.linear.x = 0.0
    pre_x, pre_z = msg.linear.x, msg.angular.z
    _cmd_vel.publish(msg)
    
if __name__ == "__main__":    
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    d=1
    # RGB Image Subscriber
    _image1 = None
    _topic_image1 = "/cam1/rgb/image_raw"
    rospy.Subscriber(_topic_image1, Image, callback_image1)
    
    # Depth Image Subscriber
    _depth1 = None
    _topic_depth1 = "/cam1/depth/image_raw"
    rospy.Subscriber(_topic_depth1, Image, callback_depth1)

    _frame = None
    rospy.Subscriber("/cam2/rgb/image_raw", Image, callback_image)

    _depth= None
    rospy.Subscriber("/cam2/depth/image_raw", Image, callback_depth)

    _cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    rospy.Subscriber("/voice/text", Voice, callback_voice)
    publisher_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)
    
    s=""
    #color
    print("hello")
    #publisher_speaker.publish("hello")
    mask = 0
    net_pose = HumanPoseEstimation()
    key = 0
    is_turning = False
    ax, ay, az, bx, by, bz = 0, 0, 0, 0, 0, 0
    pre_x, pre_z = 0.0, 0.0
    t=3.0
    '''
    '''
    haiya=0
    open_gripper(t)
    
    Kinda = np.loadtxt(RosPack().get_path("mr_dnn") + "/Kinda.csv")
    dnn_yolo = Yolov8("bagv3")
    dnn_yolo.classes = ['obj']
    '''
    chassis = RobotChassis()
    chassis.set_initial_pose_in_rviz()
    
    pos = [-1.49, 8.48, 0.00247]
    '''
    step="b"
    px, py, pz = 0,0,0
    
    l_x1,l_y1,l_x2,l_y2 = 0,0,0,0
    r_x1,r_y1,r_x2,r_y2 = 0,0,0,0
    cnt_list=[]
    jayden=0
    pre_s=""
    publisher_speaker.publish("start")
    h,w,c = _image1.shape
    img = np.zeros((h,w*2,c),dtype=np.uint8)
    img[:h,:w,:c] = _image1
    img[:h,w:,:c] = _frame
    bruh=0
    lcnt=0
    rcnt=0
    ee=""
    while not rospy.is_shutdown():
        #say("niger niger")
        t = 3.0
        if s!="" and s!=pre_s:
            print(s)
            pre_s = s
        rospy.Rate(10).sleep()
        if _frame is None:
            continue
        if _depth1 is None:
            continue
        if _image1 is None:
            continue
        if _depth is None:
            continue
        #print(ax, ay, az, bx, by, bz)
        flag = None
        depth = _depth1.copy()

        min1 = 99999999
        
        rgb_image = _frame.copy()
        rgb_image=cv2.flip(rgb_image, 0)
        frame = rgb_image.copy()
        f=_image1.copy()
        ys_no=0
        cx1,cx2,cy1,cy2=0,0,0,0
        
        if step == "b" or step=="get" or "go" in s:
            ys_no=1
            detections = dnn_yolo.forward(rgb_image)[0]["det"]
            if len(detections)>=1:
                if int(detections[0][5]) == 0:
                    detection=detections[0]
                    cnt_list=[]
                    x1,y1, x2, y2, score, class_id = map(int, detection)
                    score = detection[4]
                    
                    if score > 0.5 and class_id == 0:
                        cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cx1 = (x2 - x1) // 2 + x1
                        cy1 = (y2 - y1) // 2 + y1
                        cnt_list.append([x1, y1,x2,y2])
                        #print(cnt_list)
                        cv2.circle(rgb_image, ((x2 - x1) // 2 + x1, (y2 - y1) // 2 + y1), 5, (255, 0,0 ), -1)
                    px, py, pz = get_real_xyz(cx1, cy1)
                    l_x1,l_y1,l_x2,l_y2=x1,y1, x2, y2
                if len(detections)>=2:
                    if int(detections[1][5]) == 0:
                        detection=detections[1]
                        x1,y1, x2, y2, score, class_id = map(int, detection)
                        score = detection[4]
                        if score > 0.5 and class_id == 0:
                            cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cx2 = (x2 - x1) // 2 + x1
                            cy2 = (y2 - y1) // 2 + y1
                            cnt_list.append([x1, y1,x2,y2])
                            #print(cnt_list)
                            cv2.circle(rgb_image, ((x2 - x1) // 2 + x1, (y2 - y1) // 2 + y1), 5, (0, 255,0 ), -1)   
                        px, py, pz = get_real_xyz(cx2, cy2)
                            
                        #print(cnt_list)
                        r_x1,r_y1,r_x2,r_y2 = x1,y1, x2, y2
                    if l_x1>r_x1:
                        d1,d2,d3,d4=r_x1,r_y1,r_x2,r_y2
                        r_x1,r_y1,r_x2,r_y2 = l_x1,l_y1,l_x2,l_y2
                        l_x1,l_y1,l_x2,l_y2=d1,d2,d3,d4

                    if step!="get":
                        say("I found both bag")
                    step="get"
        if "get" in s or step == "get":
            if bruh == 0:
                say("I gonna take it now")
                bruh = 1
            cnt = 1
            pose=None
            poses = net_pose.forward(f)
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
            sx1,sx2,sx3,sx4=0,0,0,0
            cx,cy=0,0
            flag=None
            #mag=Twist()
            if pose is not None:
                f=pose_draw(f)
                msg = Twist()
                time.sleep(10)
                if bx>0:
                    haiya=1
                    cx,cy= (l_x2 - l_x1) // 2 + l_x1, (l_y2 - l_y1) // 2 + l_y1
                    sx1,sx2,sx3,sx4=l_x1,l_y1,l_x2,l_y2
                    say("left")
                    ee="left"
                elif bx<0:
                    haiya=1
                    cx,cy = (r_x2 - r_x1) // 2 + r_x1, (r_y2 - r_y1) // 2 + r_y1
                    sx1,sx2,sx3,sx4=r_x1,r_y1,r_x2,r_y2
                    say("right")
                    ee="right"
                #print(rcnt, lcnt)
                if haiya ==1:
                    rx,ry,d=get_real_xyz(cx,cy)
                    if(d==0):
                        continue
                    print(d)
                    angle = math.atan(rx / d)
                    angle_degrees = math.degrees(angle)
                    r_angle=angle*180/3.14
                    print(r_angle)
                    if r_angle > 0:
                        dz = 0.015
                    elif r_angle<0:
                        dz = -0.015
                    angle_cnt=abs(r_angle)/dz
                    for i in range(int(angle_cnt)):
                        msg.angular.z = pre_z + dz
                        pre_z=msg.angular.z
                        _cmd_vel.publish(msg)
                        h,w,c = _image1.shape
                        f=cv2.line(f, (320,0), (320,500), (0,255,0), 5)
                        rgb_image=cv2.line(rgb_image, (320,0), (320,500), (0,255,0), 5)
                        img = np.zeros((h,w*2,c),dtype=np.uint8)
                        img[:h,:w,:c] = f
                        img[:h,w:,:c] = rgb_image
                        cv2.imshow("image", img)
                    break
        if haiya != 1:
            h,w,c = _image1.shape
            f=cv2.line(f, (320,0), (320,500), (0,255,0), 5)
            rgb_image=cv2.line(rgb_image, (320,0), (320,500), (0,255,0), 5)
            img = np.zeros((h,w*2,c),dtype=np.uint8)
            img[:h,:w,:c] = f
            img[:h,w:,:c] = rgb_image
        
            cv2.imshow("image", img)

        #cv2.imshow("image", img)
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            break

