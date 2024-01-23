#!/usr/bin/env python3.8
import rospy
from sensor_msgs.msg import Image

from cv_bridge import CvBridge
import cv2
import numpy as np

from std_msgs.msg import String

from mr_voice.msg import Voice

from pcms.openvino_models import HumanPoseEstimation, FaceDetection, AgeGenderRecognition

from RobotChassis import RobotChassis
import mediapipe as mp
from geometry_msgs.msg import Twist
import math
import time

# from tf.transformations import euler_from_quaternion

# from pcms.pytorch_models import *

def callback_character(msg):
    global character
    character = msg.data


def callBack_image(msg):
    global _image
    _image = CvBridge().imgmsg_to_cv2(msg, "bgr8")


def say(text):
    global _pub_speaker
    if text is None: return
    if len(text) == 0: return
    rospy.loginfo("ROBOT: %s" % text)
    _pub_speaker.publish(text)
    rospy.sleep(1)


'''
def imu_callback(msg):
    global imu
    imu = msg
'''


def callback_depth(msg):
    global depth
    depth = CvBridge().imgmsg_to_cv2(msg, "passthrough")


def callback_voice(msg):
    global _voice
    _voice = msg


def get_real_xyz(x, y):
    global depth
    if depth is None:
        return -1, -1, -1
    h, w = depth.shape[:2]
    d = depth[y][x]
    a = 49.5 * np.pi / 180
    b = 60.0 * np.pi / 180
    real_y = (h / 2 - y) * 2 * d * np.tan(a / 2) / h
    real_x = (w / 2 - x) * 2 * d * np.tan(b / 2) / w
    return real_x, real_y, d
def getDepth(cx, cy):
    d = depth[cy][cx]
    if d == 0:
        for i in range(1, 2, 1):
            for x in range(0 - i, 0 + i, i):
                for y in range(0 - i, 0 + i, i):
                    d = depth[y][x]
                    if d != 0:
                        return d
    return d

def get_target_d(frame):
    poses = dnn_human_pose.forward(frame)
    frame = dnn_human_pose.draw_poses(frame, poses, 0.1)
    global _image
    image = _image.copy()
    x1, y1, x2, y2 = 1000, 1000, 0, 0
    d = 10000
    ti =0
    if len(poses) != 0:
        for i in range(len(poses)):
            xi, yi, c = map(int, poses[i][0])
            di = getDepth(xi,yi)
            if di!=0 and di< areaD:
                if di<d: 
                    d = di
                    ti = i
        pose = poses[ti]
        for i, p in enumerate(pose):
            x, y, c = map(int, p)
            if x < x1 and x != 0: x1 = x
            if x > x2: x2 = x
            if y < y1 and y != 0: y1 = y + 5
            if y > y2: y2 = y
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        # rospy.loginfo(cx,cy)
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
        # return cx, image
        d = depth[cy][cx]
        a = 0
        if d != 0:
            a = max(int(50 - (abs(d - 1150) * 0.0065)), 20)
        rospy.loginfo(a)
        print("d : " + str(d))
        cv2.rectangle(image, (x1, y1 - a), (x2, y2), (255, 0, 0), 2)
        if d == -1:
            return -1, -1, -1
        return d, image, [x1, y1, x2, y2]
    return -1, -1, -1


def angular_PID(cx, tx):
    e = tx - cx
    p = 0.0015
    z = p * e
    if z > 0:
        z = min(z, 0.3)
        z = max(z, 0.01)
    if z < 0:
        z = max(z, -0.3)
        z = min(z, -0.01)
    return z


def linear_PID(cd, td):
    e = cd - td
    p = 0.00025
    x = p * e
    if x > 0:
        x = min(x, 0.2)
        x = max(x, 0.1)
    if x < 0:
        x = max(x, -0.2)
        x = min(x, -0.1)
    return x

def move_status():
    while not rospy.is_shutdown():
        code = chassis.status_code
        text = chassis.status_text

        if code == 0:  # No plan.
            pass
        elif code == 1:  # Processing.
            pass
        elif code == 3:  # Reach point.
            say("I am arrived.")
            #_status = 2
            break
        elif code == 4:  # No solution.
            say("I am trying to move again.")
            break
        else:
            rospy.loginfo("%d, %s" % (code, text))
            break

def count_color(frame):
    h, w, c = frame.shape
    c = 0
    for x in range(w):
        for y in range(h):
            if frame[y, x, 0] != 0 and frame[y, x, 1] != 0 and frame[y, x, 1] != 0:
                c += 1
    return c

def detect_color(frame):
    _frame = cv2.resize(frame, (30, 40))
    hsv_frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2HSV)
    clist = []

    low_red = np.array([156, 43, 46])
    high_red = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    red = cv2.bitwise_and(_frame, _frame, mask=red_mask)
    clist.append([count_color(red), "red"])

    low_orange = np.array([5, 75, 0])
    high_orange = np.array([21, 255, 255])
    orange_mask = cv2.inRange(hsv_frame, low_orange, high_orange)
    orange = cv2.bitwise_and(_frame, _frame, mask=orange_mask)
    clist.append([count_color(orange), "orange"])

    low_yellow = np.array([22, 93, 0])
    high_yellow = np.array([33, 255, 255])
    yellow_mask = cv2.inRange(hsv_frame, low_yellow, high_yellow)
    yellow = cv2.bitwise_and(_frame, _frame, mask=yellow_mask)
    clist.append([count_color(yellow), "yellow"])
    # Green color
    low_green = np.array([34, 0, 0])
    high_green = np.array([94, 255, 255])
    green_mask = cv2.inRange(hsv_frame, low_green, high_green)
    green = cv2.bitwise_and(_frame, _frame, mask=green_mask)
    clist.append([count_color(green), "green"])

    low_blue = np.array([94, 10, 2])
    high_blue = np.array([126, 255, 255])
    blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
    blue = cv2.bitwise_and(_frame, _frame, mask=blue_mask)
    clist.append([count_color(blue), "blue"])

    low_purple = np.array([130, 43, 46])
    high_purple = np.array([145, 255, 255])
    purple_mask = cv2.inRange(hsv_frame, low_purple, high_purple)
    purple = cv2.bitwise_and(_frame, _frame, mask=purple_mask)
    clist.append([count_color(purple), "purple"])

    low_pink = np.array([143, 43, 46])
    high_pink = np.array([175, 255, 255])
    pink_mask = cv2.inRange(hsv_frame, low_pink, high_pink)
    pink = cv2.bitwise_and(_frame, _frame, mask=pink_mask)
    clist.append([count_color(pink), "pink"])

    lower_black = np.array([0,0,0])
    upper_black = np.array([180,255,50])
    black_mask = cv2.inRange(hsv_frame, lower_black, upper_black)
    black = cv2.bitwise_and(_frame, _frame, mask=black_mask)
    clist.append([count_color(black), "black"])

    lower_white = np.array([0, 0, 90])
    upper_white = np.array([172, 111, 255])
    mask = cv2.inRange(hsv_frame, lower_white, upper_white)
    white = cv2.bitwise_and(_frame, _frame, mask=mask)
    clist.append([count_color(white), "white"])

    lower_gray = np.array([0, 0, 50])
    upper_gray = np.array([255, 50, 150])
    mask = cv2.inRange(hsv_frame, lower_gray, upper_gray)
    gray = cv2.bitwise_and(_frame, _frame, mask=mask)
    clist.append([count_color(gray), "gray"])

    print(sorted(clist, reverse=True))
    return sorted(clist, reverse=True)[0][1]


def detect_cloth_color(image):
    global status
    if status != 1: return -1,-1
    Upcolor, dncolor=-1,-1
    poses = dnn_human_pose.forward(image)
    
    if len(poses) > 0:
        d = 10000
        ti =0
        if len(poses) != 0:
            for i in range(len(poses)):
                xi, yi, c = map(int, poses[i][0])
                di = getDepth(xi,yi)
                if di!=0 and di< areaD:
                    if di<d: 
                        d = di
                        ti = i
            pose = poses[ti]
        if checkupColor == True:
            Upcolor = -1
        else:
            x1 = int(pose[6][0])
            y1 = int(pose[6][1])
            x2 = int(pose[11][0])
            y2 = int(pose[11][1])
            cv2.circle(image, (x1, y1), 5, (255, 0, 0), -1)
            cv2.circle(image, (x2, y2), 5, (255, 0, 0), -1)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            # print(cx,cy)
            d = getDepth(cx, cy)
            # print("d : ",str(d))
            # print(f"x1:{x1},y1:{y1},x2:{x2},y2:{y2}")
            if (x1 != 0 and x2 != 0 and y1 != 0 and y2 != 0) and (x1 < x2 and y1 < y2) and (
                    x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0):
                # print(x1,x2,y1,y2)
                x1 -= int(d * 0.01)
                x2 += int(d * 0.01)
                cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # print(f"x1:{x1},y1:{y1},x2:{x2},y2:{y2}")
                frame = image[y1:y2, x1:x2, :]
                # cv2.imshow("up",frame)
                Upcolor = detect_color(frame)
                print("upC :", str(Upcolor))
        if checkdnColor == True:
            dncolor = -1
        else:
            x1 = int(pose[12][0])
            y1 = int(pose[12][1])
            x2 = int(pose[13][0])
            y2 = int(pose[13][1])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            d = getDepth(cx, cy)
            if (x1 != 0 and x2 != 0 and y1 != 0 and y2 != 0) and (x1 < x2 and y1 < y2) and (
                    x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0):
                # print(x1,x2,y1,y2)
                cv2.rectangle(image, (x1 - int(d * 0.015), y1), (x2 + int(d * 0.015), y2), (0, 255, 0), 2)
                # print(f"x1:{x1},y1:{y1},x2:{x2},y2:{y2}")
                frame = image[y1:y2, x1:x2, :]
                cv2.imshow("down", frame)
                dncolor = detect_color(frame)
                print("dpwnC :", str(dncolor))
                # cv2.imwrite(image,"/home/pcms/Desktop/detectColor.png")
        return Upcolor, dncolor
    return -1, -1


def getDepth(cx, cy):
    d = depth[cy][cx]
    if d == 0:
        for i in range(1, 2, 1):
            for x in range(0 - i, 0 + i, i):
                for y in range(0 - i, 0 + i, i):
                    d = depth[y][x]
                    if d != 0:
                        return d
    return d


def getMask(img):
    global status
    if status != 1: return -1
    if image is None: return -1
    h, w, c = image.shape
    if h == 0 or w == 0 or c == 0:
        return None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    upperb = np.array([54, 255, 255])
    lowerb = np.array([0, 20, 0])
    mask = cv2.inRange(img, lowerb=lowerb, upperb=upperb)
    masked = cv2.bitwise_and(img, img, mask=mask)
    return (mask, masked)


def check_mask(face_img) -> bool:
    global status
    if status != 1: return None
    if checkMask == True:return None
    face_img = face_img[int(face_img.shape[0] / 2):].copy()
    # check if it wearing mask
    face_img = cv2.resize(face_img, (224, 112))
    mask, masked = getMask(face_img)
    # get total mask pixel
    print(f"mask:{mask}")
    if mask is None:
        return None
    tot_pixel = np.sum(mask == 255)
    return tot_pixel < 9500

def get_face_img(frame):
    global status
    if status != 1: return None
    faces = dnn_face.forward(frame)
    x1,y1,x2,y2 = 0,0,1000,1000
    d = 1000000 
    if len(faces) > 0:
        for xi1,yi1,xi2,yi2 in faces:
            di = getDepth((xi1+xi2)//2,(yi1+yi2)//2)
            if di < d and di != 0:
                x1,y1,x2,y2 = xi1,yi1,xi2,yi2
        if (x1 == 0) and (y1 == 0) and (x2 == 0) and (y2 == 0): return None 
        if x1 > x2: x1,x2 = x2,x1
        if y1 > y2: y1,y2 = y2,y1
        if (x1 != 0 and x2 != 0 and y1 != 0 and y2 != 0) and (x1 < x2 and y1 < y2) and (
                x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0):
            h, w, c = frame.shape
            if h != 0 and w != 0 and c != 0:
                face_img = frame[y1:y2, x1:x2, :]
                return face_img
    return None


def detectGlasses(frame):
    global status
    if status != 1: return False
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w, c = hsv_frame.shape
    cnt = 0
    if h != 0 and w != 0 and c != 0:
        for x in range(w):
            for y in range(h):
                if hsv_frame[y, x, 2] <= 60:
                    cnt += 1
    if cnt >= 5:
        return True
    return False


def getGlasses(frame):
    global status
    if status != 1: return False
    if checkglasses == True:return False
    if frame is None: return False
    img = frame.copy()
    
    # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(img)
    x1, y1, x2, y2 = 0, 0, 0, 0
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(image=img, landmark_list=faceLms, connections=mpFaceMesh.FACEMESH_TESSELATION,
                                  connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        for id, lm in enumerate(faceLms.landmark):
            # print(lm)
            ih, iw, ic = img.shape
            if id == 55:
                x1, y1 = int(lm.x * iw), int(lm.y * ih)
            elif id == 412:
                x2, y2 = int(lm.x * iw), int(lm.y * ih)
    if x1 != 0 and y1 != 0 and x2 != 0 and y2 != 0:
        glasses_img = frame[y1:y2, x1:x2, :]
        # glasses_img = cv2.resize(glasses_img, (320, 240))
        if detectGlasses(glasses_img):
            # cv2.imshow("glasses_img", glasses_img)
            print("yessssssssssssssss")
            return True
        return False

def checkFaces(frame,cnt):
    faces = dnn_face.forward(frame)
    x1,y1,x2,y2 = -1,-1,-1,-1
    if cnt == 0:
        x1 = 1000
    elif cnt == 1:
        x1 =0
    elif cnt == 2:
        x1=1000
    if len(faces) == 0:
        return -1,-1,-1,-1
    for xi1, yi1, xi2, yi2 in faces:
        cx,cy = int((xi1+xi2)//2),int((yi1+yi2)//2)
        d = depth[cy][cx]
        if cnt == 0 and d < areaD:
            if xi1 < x1:
                x1,y1,x2,y2 = xi1, yi1, xi2, yi2
        elif cnt == 1 and d < areaD:
            if xi1> x1:
                x1,y1,x2,y2 = xi1, yi1, xi2, yi2
        elif cnt ==2 and d < areaD:
            if abs(xi1-320) < abs(x1-320):
                x1,y1,x2,y2 = xi1, yi1, xi2, yi2
                
    if x1 == 1000:
        return -1,-1,-1,-1
    else:
        return x1,y1,x2,y2

if __name__ == "__main__":
    rospy.init_node("task")
    rospy.loginfo("started task")
    _voice = None
    rospy.Subscriber("/voice/text", Voice, callback_voice)
    dnn_human_pose = HumanPoseEstimation(device_name="GPU")
    dnn_age_gender = AgeGenderRecognition(device_name="GPU")
    print("finish dnn")
    _image = None
    rospy.Subscriber("/cam1/rgb/image_raw", Image, callBack_image)
    rospy.wait_for_message("/cam1/rgb/image_raw", Image)
    print("finish rgb")

    character = None

    depth = None
    rospy.Subscriber("/cam1/depth/image_raw", Image, callback_depth)
    print("finish camera")
    path_openvino = "/home/pcms/models/openvino/"
    dnn_face = FaceDetection(path_openvino)
    #dnn_appearance = PersonAttributesRecognition(path_openvino)
    print("wait for opening yolo")
    _pub_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)
    msg_cmd = Twist()
    pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    print("readys")

    tlist = []  # appearance
    alist = []  # again appearance
    Llist = []  # location
    status = -1
    now_person = ""
    t0 = None
    chassis = RobotChassis()
    cnt = 0
    guest_color = None
    pos = {"roomL": (3.95, -0.934, -1.7), "roomR": (4.64, -0.623, -3.5),"roomM": (4.64, -0.623, -1.95), "master": (5.05, 0.937, 0.85)}
    pos_1 = {"roomL" : (-6.67,-6,-3.2),"roomR" : (-6.67,-6,0.2),"roomM": (-6.67,-6,-1.75),"master" : (-6.67,-3.76,1.7)}
    pos_Fablab ={"roomL" : (6.23,-1.5,-0.3),"roomR" : (6.23,-1.5,3),"roomM": (6.23,-1.5,-1.2),"roomB": (6.23,-1.5,2),"master_wait" :(6.24,-1.81,3),"master" : (-0.624,-4.62,-1.6),"DWP_C":(-0.169,-2.2,2),"GWP_C" : (0.827,-0.678,0),"DWP_B":(-0.169,-2.2,-1.6),"GWP_B" : (0.827,-0.678,3)}
    #angList_Fablab = {"Left" : {"R1":20,"R2":50,"D1":65,"D2":90,"BP1" : 120,"BP2":150,"C1":215,"C2":235},"Right" :{"BP1":-30,"BP2":-75,"D1":-80,"D2":-110,"R1":-135,"R2":-145,"C1":-260,"C2":-280},"Middle1" :{"D1":0,"D2":-40},"Middle2":{"C1":30,"C2":-30}}
    angList_Fablab = {"Left" : {"S":[-35,0,"sink"],"M":[0,30,"washing machine"],"F" : [30,45,"refrigerator"],"TV":[46,90,"TV"],"R":[90,130,"rubbish bin"],"D":[130,145,"door"],"C":[160,260,"chair"],"CB":[270,315,"cabinet"]},"Right" :{"CB":[35,0,"cabinet"],"C":[0,-125,"chair"],"D":[-130,-140,"door"],"R":[-140,-180,"rubbish bin"],"TV":[-180,-225,"TV"],"F":[-225,-240,"refrigerator"],"M":[-240,-270,"washing machine"],"S":[-270,-315,"sink"]},"Middle" :{"D":[-10,10,"door"],"R":[-25,-10,"rubbish bin"],"TV":[-45,-25,"TV"],"C":[10,45,"chair"]}}
    RCJPos = {"roomL" : (4.62,9,-0.75),"roomR" : (4.1,9.19,0.15),"roomM": (4.62,9.19,-2.2),"master" : (7.79,11.1,2.4),"come_wait" :(7.39,8.54,3),"master_wait" :(6.5,8.54,0),"CL" :(3.2,10.5,1.05),"CR":(5.38,11.6,3.14)}   
    angList =[]
    areaD = 2700
    c = 0

    # status =1
    
    have_glasses = False
    publish = False
    save = False
    yolo = False
    publish = False
    hasMask = -1
    upcolor = -1
    dncolor = -1
    gender = -1
    checkupColor = False
    checkdnColor = False
    checkMask = False
    checkglasses = False
    checkGender = False

    NotP = False
    go = ""

    angular = 0
    i=0
    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
    mp_drawing_styles = mp.solutions.drawing_styles

    while not rospy.is_shutdown():
        rospy.Rate(30).sleep()
        msg_cmd.linear.x = 0.0
        msg_cmd.angular.z = 0.0
        tlist = []
        # cv2.imshow("_image",_image)
        frame = _image.copy()
        print(status)
        if status == -1:  # go to the room with guests in it
            print(cnt)
            if cnt == 0:
                if _voice is None: continue
                if "living" in _voice.text and "room" in _voice.text:
                    rospy.loginfo(_voice.text)
                    say("ok")
                    chassis.move_to(*RCJPos["come_wait"])
                    #chassis.move_to(*pos["door"])
                    chassis.move_to(*RCJPos["roomL"])
                    say('I arrived living room')
                    _voice = None
                    status +=1
                    print("I arrived outside the room")
                    angular = 0
                    rospy.sleep(0.5)
                    
            elif cnt == 1:
                #say("ok")
                chassis.move_to(*RCJPos["come_wait"])
                chassis.move_to(*RCJPos["roomR"])
                say('I arrived living room')
                _voice = None
                status +=1
                print("I arrived outside the room")
                angular = 0
                rospy.sleep(0.5)
            else:
                #say("ok")
                chassis.move_to(*RCJPos["come_wait"])
                chassis.move_to(*RCJPos["roomM"])
                say('I arrived living room')
                _voice = None
                status +=1
                print("I arrived outside the room")
                angular = 0
                rospy.sleep(0.5)

        elif status == 0:
            x1,y1,x2,y2 = checkFaces(frame,cnt)
            if x1 != -1:
                save_frame = frame.copy()
                cv2.rectangle(save_frame,(x1,y1),(x2,y2),(0, 255, 0), 2)
                cv2.imwrite(f"/home/pcms/Desktop/detect_guest_{str(cnt)}.png",save_frame)
                cx = ((x1 + x2) // 2)
                if max(cx, 315) == cx and min(cx, 325) == cx:
                    
                    print("finish")
                    msg_cmd.angular.z = 0.0
                    save = False
                    yolo = False
                    publish = False
                    rospy.sleep(0.05)
                    hasMask = -1
                    upcolor = -1
                    dncolor = -1
                    gender = -1
                    i = 0
                    have_glasses = False
                    status+=1
                    say("please wait")
                    
                else:
                    v = angular_PID(cx, 320)
                    msg_cmd.angular.z = v
                    angular += math.ceil(v *12.5*-1)
            else:
                if cnt == 0:
                    msg_cmd.angular.z = -0.2
                    angular += 1.5
                elif cnt == 1:
                    msg_cmd.angular.z = 0.2
                    angular -= 1.5

            print(f"angular:{angular}") 
            print(msg_cmd.angular.z)
            pub_cmd.publish(msg_cmd)

        elif status == 1:
            if frame is None: continue
            face = frame.copy()
            d, image, clist = get_target_d(frame)
            face_img = get_face_img(face)

            if clist == -1: continue
            _frame = _image.copy()
            # print(appearance)
            cv2.imshow("test", image)
            if checkupColor == False or checkdnColor == False:
                bodyPart = frame[clist[1] - 15:clist[3] + 15, clist[0]:clist[2], :]  # y1:y2,x1:x2
                # cv2.imshow("body",bodyPart)
                upcolor, dncolor = detect_cloth_color(frame)
            if face_img is not None and checkMask == False:
                hasMask = check_mask(face_img)
            print(f"hasMask:{hasMask}")
            if face_img is not None and checkGender == False:
                age, gender = dnn_age_gender.forward(face_img)
                gender = dnn_age_gender.genders_label[gender]
            if i < 5 and checkglasses == False:
                have_glasses = have_glasses or getGlasses(face_img)
                print(f"have_glass:{have_glasses}")
                i += 1
            if checkupColor == False:
                if upcolor == -1:
                    continue
            if checkdnColor == False:
                if dncolor == -1:
                    continue
            if checkMask == False:
                if hasMask is None:
                    continue
            if checkglasses == False:
                if i < 5:
                    continue
            if checkGender == False:
                if gender == -1:
                    continue
            if NotP == True:
                if go  == "Left":
                    chassis.move_to(*RCJPos["CL"])
                elif go =="Right":
                    chassis.move_to(*RCJPos["CR"])
                status=1.5
            else:
                status += 1
            i = 0
            print(f"gender:{gender}")
            say("I am approaching")
            cv2.imshow("faceImage", face_img)

        elif status==1.5:
            x1,y1,x2,y2 = checkFaces(frame,cnt)
            if x1 != -1:
                save_frame = frame.copy()
                cv2.rectangle(save_frame,(x1,y1),(x2,y2),(0, 255, 0), 2)
                cv2.imwrite(f"/home/pcms/Desktop/detect_guest_{str(cnt)}.png",save_frame)
                cx = ((x1 + x2) // 2)
                if max(cx, 315) == cx and min(cx, 325) == cx:
                    status += 1
                    print("finish")
                    msg_cmd.angular.z = 0.0
                    rospy.sleep(0.05)
                    i = 0
                    have_glasses = False
                    status=2
                    say("please wait")
                else:
                    v = angular_PID(cx, 320)
                    msg_cmd.angular.z = v
            
        elif status == 2:
            if NotP == True: 
                status += 1
                continue
            if frame is None: continue
            print("frame is exist")
            d, image, clist = get_target_d(frame)
            # bodyPart = frame[clist[1]:clist[3], clist[0]:clist[2], :]  # y1:y2,x1:x2 
            if d != -1 and d != None:
                print("d is not None")
                if d == 0:
                    print("d == 0")
                    if clist != -1:
                        cx = int((clist[0] + clist[2])) // 2
                        cy = int((clist[1] + clist[3])) // 2
                        d = getDepth(cx, cy)
                rospy.loginfo(d)
                if d < 800 or d > 900:
                    print("executing PID Function!")
                    v = linear_PID(d, 850)
                    
                    msg_cmd.linear.x = v
                    print(f"speed : {v}")
                else:
                    #rospy.sleep(0.05)
                    for i in range(7):
                        msg_cmd.linear.x = 0.2 
                        pub_cmd.publish(msg_cmd)
                    msg_cmd.linear.x = 0.0
                    rospy.sleep(1)
                    say('I found a guest')
                    status += 1
                    print("done")
                    #rospy.sleep(0.05)
                    say("What is your name")
                    print("say your name")
                    
                frame = image
            pub_cmd.publish(msg_cmd)

        elif status == 3:
            if _voice is None: continue
            Jameslist = ["same", "dreams","James","Maria"]
            Alexanderlist = ["panda","under","Alexander","Amanda"]
            Georgelist = ["storadge","George","joy"]
            Henrylist = ["Stanley","Henry"]
            Daniellist = ["new","Daniel"]
            Hunterlist = ["Sonta","Santa","Hunter"]
            Emmalist = ["Emma"]
            Johnlist = ["John","join","drawing","lord"]
            Oilverlist = ["Oilver","Elsa"]
            Sophialist = ["lower","Sophia"]
            Thomaslist = ["Thomas"]
            Williamlist = ["William", "Lily","volume"]
            rospy.loginfo(_voice.text)
            v = _voice.text.split(" ")[-1]
            if v == "is" or v == "name" or (v not in Jameslist and v not in Georgelist and v not in Henrylist and v not in Alexanderlist and v not in Daniellist and v not in Hunterlist and v not in Emmalist and v not in Johnlist and v not in Oilverlist and v not in Sophialist and v not in Thomaslist and v not in Williamlist):
                say("Could you repeat your name")
                _voice = None
                continue 
            elif v in Jameslist:
                now_person = "James"
            elif v in Alexanderlist:
                now_person = "Alexander"
            elif v in Georgelist:
                now_person = "George"
            elif v in Henrylist:
                now_person = "Henry"
            elif v in Daniellist:
                now_person = "Daniel"
            elif v in Hunterlist:
                now_person = "Hunter"
            elif v in Emmalist:
                now_person = "Emma"
            elif v in Johnlist:
                now_person = "John"
            elif v in Oilverlist:
                now_person = "Oilver"
            elif v in Thomaslist:
                now_person = "Thomas"
            elif v in Williamlist:
                now_person = "William"
            rospy.loginfo(now_person)
            _voice = None
            status += 1
            
        elif status == 4:
            #chassis.move_to(*pos["door"])
            #chassis.move_to(*pos["waitpoint1_back"])
            say("ok, I am going back now")
            chassis.move_to(*RCJPos["master_wait"])
            chassis.move_to(*RCJPos["master"])
            #chassis.move_to(*RCJPos["roomM"])
            #chassis.move_to(*pos_Fablab["master"])
            #move_status()
            #print(character)
            #while character is None:
            #    rospy.Rate(20).sleep()
            #    continue
            tem = []
            #print("character : " + character)
            # imagePath.publish(None)
            # tlist.append(character)
            #gender = character.split()[1]
            tlist = []
            print(f"gender:{gender}")
            say(f"I found {now_person} in the room")
            if hasMask and "mask" not in alist and len(tlist) < 2:
                tlist.append(f"{now_person} is wearing a mask")
                alist.append("mask")
                checkMask = True
            if have_glasses and "glass" not in alist and len(tlist) < 2:
                tlist.append(f"{now_person} is wearing a glasses")
                alist.append("glass")
                checkglasses = True
            if "upcolor" not in alist and len(tlist) < 2:
                tlist.append(f"{now_person} is wearing a {upcolor} cloth")
                alist.append("upcolor")
                checkupColor = True
            if "dncolor" not in alist and len(tlist) < 2:
                tlist.append(f"{now_person} is wearing a lower cloth in {dncolor}")
                alist.append("dncolor")
                checkdnColor = True
            if "gender" not in alist and len(tlist) < 2 and cnt == 2:
                tlist.append(f"{now_person} is a {gender}")
                alist.append("gender")
            if "hair" not in alist and len(tlist) < 2 and cnt == 2:
                if gender == "male" or gender == "boy":
                    tlist.append(f"{now_person} has short hair")
                else:
                    tlist.append(f"{now_person} has long hair")
                alist.append("hair")
            for t in tlist:
                say(t)
                #rint(f"t:{t}")
                rospy.sleep(0.05)
            angular = int(angular)
            if cnt == 0:
                say(str(now_person) + " is next to a refrigerator")
                '''
                if angular <= 45:
                    say(f"{now_person} is next to a book")
                elif angular > 45 and angular <= 90:
                    say(f"{now_person} is next to a box")
                elif angular > 90 and angular <= 135:
                    say(f"{now_person} is next to a box")
                else:
                for k,v in angList_Fablab["Left"].items():
                    if angular >= v[0] and angular <= v[1]:
                        say(str(now_person) + " is next to a " + v[2])
                        break'''
                '''        
                if angular > angList_Fablab["Left"]["S1"] and angular < angList_Fablab["Left"]["S2"]:
                    say(f"{now_person} is next to a sink")
                elif angular > angList_Fablab["Left"]["M1"] and angular < angList_Fablab["Left"]["M2"]:
                    say(f"{now_person} is next to a machine")
                elif angular >angList_Fablab["Left"]["F1"] and angular < angList_Fablab["Left"]["BP2"]:
                    say(f"{now_person} is next to a refrigerator")
                elif angular >angList_Fablab["Left"]["TV1"] and angular < angList_Fablab["Left"]["TV2"]:
                    say(f"{now_person} is next to a TV")
                elif angular >angList_Fablab["Left"]["R1"] and angular < angList_Fablab["Left"]["R2"]:
                    say(f"{now_person} is next to a rubbish bin")
                elif angular >angList_Fablab["Left"]["D1"] and angular < angList_Fablab["Left"]["D2"]:
                    say(f"{now_person} is next to a door")
                elif angular >angList_Fablab["Left"]["C1"] and angular < angList_Fablab["Left"]["C2"]:
                    say(f"{now_person} is next to a chair")
                elif angular >angList_Fablab["Left"]["CB1"] and angular < angList_Fablab["Left"]["CB2"]:
                    say(f"{now_person} is next to a cupboard")  '''
            elif cnt == 1:
                say(str(now_person) + " is next to a cabinet")
                #for k,v in angList_Fablab["Right"].items():
                #    if angular <= v[0] and angular >= v[1]:
                #        say(str(now_person) + " is next to a " + v[2])
                #        break
                '''if angular <angList_Fablab["Right"]["CB1"] and angular > angList_Fablab["Left"]["BP2"]:
                    say(f"{now_person} is next to a transparent plastic box")
                elif angular < angList_Fablab["Right"]["D1"] and angular > angList_Fablab["Right"]["D2"]:
                    say(f"{now_person} is next to a door")
                elif angular < angList_Fablab["Right"]["R1"] and angular > angList_Fablab["Right"]["R2"]:
                    say(f"{now_person} is next to a rubbish bin")
                elif angular <angList_Fablab["Right"]["C1"] and angular > angList_Fablab["Right"]["C2"]:
                    say(f"{now_person} is next to a table which has a cartons")'''
            elif cnt ==2:
                say(str(now_person) + " is next to a TV")
                '''if cb == True:
                    if angular > angList_Fablab["Middle1"]["D2"] and angular < angList_Fablab["Middle1"]["D1"]:
                        say(f"{now_person} is next to the door")
                else:
                    if angular > angList_Fablab["Middle2"]["C2"] and angular < angList_Fablab["Middle2"]["C1"]:
                        say(f"{now_person} is next to a table which has a cartons")'''
            print(f"angular: {angular}")
            
            angular = 0
            print("status :", status)
            if cnt > 2:
                break
            print("waiting for new comment")
            time.sleep(3)
            status = -1
            cnt += 1
        cv2.imshow("image", frame)
        cv2.waitKey(1)