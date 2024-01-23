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

fablab = {"door":(-6.88,-4.83,-1.54),"seat1":(-6.36,-2.86,-4.89),"seat2":(-5.52,-3.01,-4.89),"master":(-4.69,-2.91,-0.18)}

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
    rospy.sleep(1)
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

def detect_seat(img):
    if img == None:return None
    items = dnn_yolo.forward(img)
    for i in items:
        x1,y1,x2,y2,score,cid = map(int, i)
        if cid == COCO_CLASSES.index("chair"):
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),5)
    return img
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
    dnn_pose = HumanPoseEstimation()
    dnn_yolo = Yolov8("yolov8n",device_name="GPU")

    print("ready")
    say("I am ready.")
    status = 1
    cnt = 0
    
    guest = []
    while not rospy.is_shutdown():
        rospy.Rate(30).sleep()
        if _frame is None:continue
        image = _frame.copy()
        if status == 1:
            poses = dnn_pose.forward(image)
            if poses is None:continue
            print("detect the person!")
            
            x1, y1, c1 = map(int, poses[0][5])
            x2, y2, c2 = map(int, poses[0][12])
            cx = (x1+x2)//2
            cy = (y1+y2)//2
            cv2.circle(image, (cx,cy), 3, (0,0,255), -1)
            d = get_depth(cx,cy)
            print(d)
            #say("I am ready.")
            cv2.imshow("image",image)
            cv2.waitKey(0)
            exit(0)
            if d < 1200:
                say("Welcome to the party")
                say("I am Fambot, what is your name")
                while _voice is None or "name" not in _voice.text:continue
                if _voice is not None:
                    name = _voice.text.split(" ")[-1]
                    _voice = None
                    say("OK. What is your favorite drink")
                    while _voice is None or "drink" not in _voice.text:
                        rospy.sleep(0.1)
                        continue
                    drink = _voice.text.split(" ")[-2:]
                    guest.append([name,drink])
            else:
                continue
            say("OK. Please follow me")
            status += 1
        elif status == 2:
            chassis.move_to(*fablab["seat1"])
            have_guest = False
            boxes = dnn_yolo.forward(image)[0]["det"]
            for box in boxes:
                x1, y1, x2, y2, score, class_id = map(int,box)
                color = (255,0,0)
                if class_id == 0:
                    have_guest = True
                    color = (0,0,255)
                elif class_id == COCO_CLASSES.index("chair"):color = (0,255,0)
                cv2.rectangle(image,(x1,y1),(x2,y2),color,5)
            cv2.imshow("image",image)
            if have_guest:
                chassis.move_to(*fablab["seat2"])
            say("Please have a seat in front of me")
            rospy.sleep(1)
            if cnt == 1:
                say("The person who sit next to you is %s. %s's favorite drink is %s" % (guest[0][0],guest[0][1]))
            chassis.move_to(*fablab["master"])
            say("%s has come. %s's favorite drink is %s" % (guest[cnt][0],guest[cnt][0],guest[cnt][1][0],guest[cnt][1][1]))
            cnt += 1
            if cnt == 2:
                break
            status = 1
            chassis.move_to(*fablab["door"])
        '''
        the point waiting to be set:
        -door
        -master
        -seat1
        -seat2
        '''