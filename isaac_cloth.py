#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import HumanPoseEstimation
import numpy as np

def callback_image(msg):
    global frame
    frame = CvBridge().imgmsg_to_cv2(msg, "bgr8")

def callback_depth(msg):
    global depth
    depth = CvBridge().imgmsg_to_cv2(msg, "passthrough")

if __name__=="__main__":
    rospy.init_node("isaac_mask")
    rospy.loginfo("isaac node init")

    # load model
    model = FaceDetection()
    rospy.loginfo("model load!")

    frame = None
    topic_name = "/cam1/rgb/image_raw"
    rospy.Subscriber(topic_name, Image, callback_image)
    rospy.wait_for_message(topic_name, Image)

    while True:
        rospy.Rate(20).sleep()
        # result: [[x1, y1, x2, y2], ...]
        result = model.forward(frame)

        # draw the rectangle
        for x1, y1, x2, y2 in result:
            face_img = frame[y1:y2, x1:x2]
            if check_mask(face_img):
                # mask, draw text
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Mask", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                 # no mask, draw text
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"No Mask", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # show the image
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break