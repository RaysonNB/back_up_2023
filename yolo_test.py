#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import Yolov8
from rospkg import RosPack
import time
import torch
import numpy as np
import datetime
def callback_image(msg):
    global _image
    _image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
def say(a): 
    publisher_speaker.publish(a) 

def callback_depth(msg):
    global _depth
    _depth = CvBridge().imgmsg_to_cv2(msg, "passthrough")



if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    _image = None
    rospy.Subscriber("camera/color/image_raw", Image, callback_image)

    dnn_yolo = Yolov8("brianbottlev1",device_name="GPU")
    dnn_yolo.classes = ['obj']
    rospy.sleep(1)
    fps, fps_n = 0, 0
    print("no while")
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        if _image is None:
            print("no_img")
            continue
        frame = _image.copy()
        detections = dnn_yolo.forward(frame)[0]["det"]
        yn="no"
        h, w = frame.shape[:2]
        for i, detection in enumerate(detections):
            print(detection)
            x1, y1, x2, y2, score, class_id = map(int, detection)
            score = detection[4]
            #print(x1, y1, x2, y2, score, class_id)
            cx = (x2 - x1) // 2 + x1
            cy = (y2 - y1) // 2 + y1
            if score > 0.5:# and class_id==39:
                step2="turn"
                #dnn_yolo.draw_bounding_box(detection, frame)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, "person", (x1+5, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        #frame = cv2.flip(frame, 0)
        cv2.imshow("frame", frame)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break
    rospy.loginfo("demo node end!")
