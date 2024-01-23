#!/usr/bin/env python3
import rospy
from mr_voice.msg import Voice
import cv2
import yaml
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from geometry_msgs.msg import Twist
from pcms.pytorch_models import *
from pcms.openvino_models import Yolov8

def callback_image(msg):
    global _frame
    _frame = CvBridge().imgmsg_to_cv2(msg, "bgr8")

def callback_voice(msg):
    global _voice
    _voice = msg

def callback_depth(msg):
    global _depth
    _depth = CvBridge().imgmsg_to_cv2(msg, "passthrough")

def text_to_cmd(s):
    global _cmds
    if _cmds is None or s is None: return None
    s = s.lower()
    for c in _cmds["QnA"]:
        OK = True
        for i in c["K"]:
            tmp = False
            for j in i:
                if str(j).lower() in s:
                    tmp = True
                    break
            if not tmp: OK = False
        if OK: return c
    return None

def find_chair():
    global min1,frame,px,py,pz,ax,ay,az,bx,by,bz
    for id, index, conf, x1, y1, x2, y2 in boxes:
        if(index == 56):
            name=ddn_rcnn.labels[index]
            if name=="chair":
                cx1 = (x2 - x1) // 2 + x1
                cy1 = (y2 - y1) // 2 + y1
                cv2.circle(frame, (cx1, cy1), 5, (0, 255, 0), -1)
                for(count):
   if(a[i] != cx1 and a[i] = 0):
        a[i] = cx1
        break
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0)
def find_people():
    global min1,frame,px,py,pz,ax,ay,az,bx,by,bz
    for id, index, conf, x1, y1, x2, y2 in boxes:
        if(index == 0):
            name=ddn_rcnn.labels[index]
            if name=="people":
                cx1 = (x2 - x1) // 2 + x1
                cy1 = (y2 - y1) // 2 + y1
                cv2.circle(frame, (cx1, cy1), 5, (0, 255, 0), -1)
   for(count1):
        if(b[i] != cx1 and b[i] = 0):
            b[i] = cx1
            break
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0)
if __name__  == "__main__":
    rospy.init_node("demo3")
    rospy.loginfo("demo3 start!")
    chassis = RobotChassis()
    _cmds = None
    ddn_rcnn = Yolov8()
   _image1 = None
    _topic_image1 = "/camera/color/image_raw"
    rospy.Subscriber(_topic_image1, Image, callback_image1)
   
    _depth1 = None
    _topic_depth1 = "/camera/depth/image_raw"
    rospy.Subscriber(_topic_depth1, Image, callback_depth1)

    _frame = None
    rospy.Subscriber("/cam2/rgb/image_raw", Image, callback_image)

    _depth= None
    a = [0,0,0,0]
    b = [0,0,0,0]
    count = 0
    count1 = 0
    rospy.Subscriber("/cam2/depth/image_raw", Image, callback_depth)
    with open("/home/pcms/catkin_ws/src/beginner_tutorials/src/cmd.txt", "r") as f:
        try:
            _cmds = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
        _voice = None
    _topic_voice = "/voice/text"
    rospy.Subscriber(_topic_voice, Voice, callback_voice)
    while not rospy.is_shutdown() and people <= 3:
        if _frame is None:
            continue
        if _depth1 is None:
            continue
        if _image1 is None:
            continue
        if _depth is None:
            continue
        rospy.Rate(20).sleep()
        if _voice is not None:
            rospy.loginfo("%s (%d)" % (_voice.text, _voice.direction))
           
            _cmd = text_to_cmd(_voice.text)
            if _cmd is not None:
                print(_cmd)
           
            _voice = None
        for(4):
            find_people()
            find_chair()
        a.sort()
        b.sort()
        for(4):
            if(abs(a[i] - b[i]) < 5):
        a[i] = 0, b[i] = 0

        for(4):
            if(a[i] != 0):
            rospy.loginfo("set in the %d chair" %(i))
            people +=1 
            break
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            break        
    rospy.loginfo("demo3 end!")
