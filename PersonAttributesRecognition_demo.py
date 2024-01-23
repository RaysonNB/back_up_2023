#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import *
from pcms.pytorch_models import *


def callback_image(msg):
    global _image
    _image = CvBridge().imgmsg_to_cv2(msg, "bgr8")


if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")

    # ROS Topics
    _image = None
    rospy.Subscriber("/camera/rgb/image_raw", Image, callback_image)
    rospy.wait_for_message("/camera/rgb/image_raw", Image)

    # PyTorch
    dnn_yolo = Yolov5()

    # OpenVINO
    dnn_attrs = PersonAttributesRecognition()

    # MAIN LOOP
    rospy.sleep(1)
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        image = _image.copy()
        frame = _image.copy()

        # OpenVINO
        boxes = dnn_yolo.forward(image)
        for id, index, conf, x1, y1, x2, y2 in boxes:
            if dnn_yolo.labels[index] != "person": continue

            person = image[y1:y2, x1:x2, :]
            attrs = dnn_attrs.forward(person)
            print(attrs)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            i = 0
            for k, v in attrs.items():
                cv2.putText(frame, "%s: %d" % (k, v), (x1 + 5, y1 + 15 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 125), 2)
                i = i + 1

        # show image
        cv2.imshow("frame", frame)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break

    rospy.loginfo("demo node end!")
