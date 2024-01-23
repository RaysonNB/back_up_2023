#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

def callback_image(msg):
    global _image
    _image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
    
if __name__=="__main__":
    rospy.init_node("demo2")
    print("start")
    _image = None
    rospy.Subscriber("/camera/color/image_raw", Image, callback_image)
    
    rospy.sleep(1)
    
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        rate.sleep()
        if _image is None: continue
        
        cv2.imshow("image", _image)
        key_code = cv2.waitKey(1)
        if key_code in [27,ord('q')]:
            break
