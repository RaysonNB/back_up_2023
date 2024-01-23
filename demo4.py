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
    
def find_chair():
    global min1,image,px,py,pz,ax,ay,az,bx,by,bz
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
            
if __name__  == "__main__":
    rospy.init_node("demo4")
    rospy.loginfo("demo4 start!")
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
    rospy.Subscriber("/cam2/depth/image_raw", Image, callback_depth)
    a = 0
    #b = [0,0,0,0]
    #count = 0
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
        image = _image1.copy()
        #for(4):
         #   find_people()
        #a.sort()
        #for(4):
         #   print(a[i])
          #  break
        print(a)
    key = cv2.waitKey(1)
    if key in [ord('q'), 27]:
        break        
rospy.loginfo("demo4 end!")

