#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
from RobotChassis import RobotChassis
from pcms.openvino_models import Yolov8,HumanPoseEstimation
from mr_voice.msg import Voice
from std_msgs.msg import String
import yaml
'''
***remember to set the name list and the drink list
***
the point waiting to be set:
-door
-master
-seat1
-seat2
-D_TO_S
-S_TO_M
-M_TO_D
'''

'''
改open name list 的路徑
在SSD問題集的路徑並根據namelist更改
在TLM上詢問座位位置
'''
COCO_CLASSES = (
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
)

fablab = {"door":(2.64,-1.16,-3.14),"seat1":(5.33,-0.451,1.57),"seat2":(5.94,-0.61,1.57),"master":(6.6,-4.9,-1.57),"m_to_d":(7.02,-2.21,1.57),"D_TO_S":(5.42,-2.06,0)}
          
with open("/home/pcms/keys.txt", "r") as f:
    try:
        _keys = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(e)

fruit = ['apple','orange']
def callback_image(msg):
    global _frame
    _frame = CvBridge().imgmsg_to_cv2(msg, "bgr8")

def callback_voice(msg):
    global _voice
    _voice = msg

def callback_depth(msg):
    global depth
    depth = CvBridge().imgmsg_to_cv2(msg,"passthrough")

def say(t):
    global _pub_speaker
    if t is None or len(t) == 0:return
    _pub_speaker.publish(t)
    rospy.loginfo(t)
    rospy.sleep(2)
    
def get_depth(cx,cy):
    global depth
    d = depth[cy][cx]
    if d == 0:
        for i in range(-1,2,1):
            for j in range(-1,2,1):
                d = depth[cy+i][cx+j]
                if d != 0:return d
    else:
        return d
    return -1

'''
def detect_seat(img):
    if img == None:return None
    items = dnn_yolo.forward(img)
    for i in items:
        x1,y1,x2,y2,score,cid = map(int, i)
        if cid == COCO_CLASSES.index("chair"):
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),5)
    return img
'''
def check_person(img):
    global dnn_yolo
    boxes = dnn_yolo.forward(img)[0]["det"]
    chairs = []
    person = None
    for i,box in enumerate(boxes):
        x1, y1, x2, y2, class_id, score = map(int, box)
        cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),5)
        cv2.putText(img, str(i), (x1,y1-5), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
        print(f"number {i}: {class_id}")
        if class_id == COCO_CLASSES.index("chair"):
            print("find a chair")
            
            chairs.append(x1)
        elif class_id == 0:
            size = abs(x2 - x1) * abs(y2-y1)
            if size / (640*480) >= 0.2:
                person = x1
    if person is None or abs(person - min(chairs)) < abs(person - max(chairs)):
        return 1, img
    else:
        print("I see a guest")
        return 2, img

def text_to_ans(s):
    s = s.lower()
    for c in _keys["QnA"]:
        OK = True
        for i in c["K"]:
            tmp = False
            for j in i:
                if str(j).lower() in s:
                    tmp = True
                    break
            if not tmp: OK = False
        if OK: return c["A"]
    return None


if __name__ == "__main__":
    rospy.init_node("task3")
    rospy.loginfo("task3 started!")
    
    _frame = None
    rospy.Subscriber("/camera/rgb/image_raw",Image,callback_image)
    
    depth = None
    rospy.Subscriber("/camera/depth/image_raw", Image, callback_depth)

    _voice = None
    rospy.Subscriber("/voice/text", Voice, callback_voice)

    _pub_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)

    chassis = RobotChassis()
    
    msg_cmd = Twist()
    pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    dnn_pose = HumanPoseEstimation(device_name="GPU")
    dnn_yolo = Yolov8("yolov8n",device_name="GPU")

    print("ready")
    status = 2
    cnt = 0

    guest = []
    while not rospy.is_shutdown():
        rospy.Rate(30).sleep()
        if _frame is None:
            print("no frame")
            continue
        image = _frame.copy()
        print(1)
        if status == 1:
            poses = dnn_pose.forward(image)
            if poses is None:continue
            print("detect the person!")
            d = 9999
            boxes = dnn_yolo.forward(image)[0]["det"]
            for box in boxes:
                x1, y1, x2, y2, score, class_id = map(int,box)
                if class_id == 0:
                    cx = (x1+x2)//2
                    cy = (y1+y2)//2
                    d = get_depth(cx,cy)
                    break            
            if d < 1300:
                say("Welcome to the party")
                say("I am Fambot, what is your name")
                while _voice is None or "name" not in _voice.text:continue

                while _voice is not None:
                    name = text_to_ans(_voice.text.split(" ")[-1])
                    if name is None:continue
                    say("you are %s" % name)
                    break
                else:
                    continue
                say("OK. What is your favorite drink")
                while _voice is None or "drink" not in _voice.text:continue
                while _voice is not None:
                    if "drink" not in _voice.text:
                        continue
                    app = text_to_ans(_voice.text)
                    print("app",app) 
                    if app is None:
                        say("could you repeat again")
                        continue
                    else:
                        _voice = None
                        say("your favorite drink is %s" % app)
                guest.append([name,app])
            else:
                continue                    
            say("OK. Please follow me")
            status += 1
        elif status == 2:
            say("step two")
            #chassis.move_to(*fablab["D_TO_S"])

            '''
            vel = 0.2
            msg_cmd.linear.x = vel
            pub_cmd.publish(msg_cmd)
            t0 = rospy.Time().now().to_sec()
            target_distance = 2.0
            current_distance = 0.0
            while target_distance > current_distance:
                t1 = rospy.Time.now().to_sec()
                current_distance = (t1-t0) * vel
            msg_cmd.linear.x = 0.0
            pub_cmd.publish(msg_cmd)
            '''

            #chassis.move_to(*fablab["seat"])

            have_guest = False
            image = _frame.copy()
            while image is None:
                image = _frame.copy()
                continue
            seat, empty_s = check_person(image)
            '''
            boxes = dnn_yolo.forward(image)[0]["det"]
            for box in boxes:
                x1, y1, x2, y2, score, class_id = map(int,box)
                cv2.rectangle(image, (x1,y1),(x2,y2),(0,255,0),5)
                cv2.imwrite("/home/pcms/Desktop/yolo_person.jpg",image)
                color = (255,0,0)
                if class_id == 0:
                    size = abs(x2 - x1) * abs(y2-y1)
                    if size / (640*480) >= 0.15:
                        have_guest = True
                        print("there is a guest in front of me")
            '''
            cv2.imwrite("/home/pcms/Desktop/empty_seat.jpg",empty_s)
            cv2.imshow("image",image)
            if seat == 1:
                #chassis.move_to(*fablab["seat1"])
                say("go to seat one")
            elif seat == 2:
                #chassis.move_to(*fablab["seat2"])
                say("go to seat two")
            #say("Please have a seat in front of me after I leave")
            break
            image = _frame.copy()
            cv2.imwrite("/home/pcms/Desktop/new_img.jpg",image)
            rospy.sleep(1)
            if cnt == 1:
                say("Let me introduce the person sits next to you. This person  is %s. %s's favorite drink is %s" % (guest[0][0],guest[0][0],guest[0][1]))
            chassis.move_to(*fablab["master"])
            say(f"{guest[cnt][0]} has come. {guest[cnt][0]}'s favorite drink is {guest[cnt][1]}")
            cnt += 1
            if cnt == 2:
                break
            while _voice is None:continue
            while "door" not in _voice.text:
                _voice = None
                while _voice is None:continue
                continue
            say("OK, I will back now")
            chassis.move_to(*fablab["m_to_d"])
            chassis.move_to(*fablab["door"])
            status = 1
