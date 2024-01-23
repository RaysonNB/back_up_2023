#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
import pyzbar.pyzbar as pyzbar
from cv_bridge import CvBridge
import numpy as np
def callback_image(msg):
    global _frame
    _frame = CvBridge().imgmsg_to_cv2(msg, "bgr8")
def say(a): 
    publisher_speaker.publish(a) 
if __name__ == "__main__": 
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    
    _frame = None
    rospy.Subscriber("/camera/rgb/image_raw", Image, callback_image)
    
    while not rospy.is_shutdown():
        rospy.Rate(10).sleep()
        if _frame is None: continue
        image=_frame.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            texts = pyzbar.decode(gray)
            print(texts)
            if texts==[]:
                print("未識別成功")
            else:
                for text in texts:
                    tt = text.data.decode("utf-8")
                print("識別成功")
                print(tt)
            print("NONE")
            cv2.imshow("frame", _frame)
            key_code = cv2.waitKey(1)
            if key_code in [27, ord('q')]:
                break


