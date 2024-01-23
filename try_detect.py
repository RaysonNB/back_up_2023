#!/usr/bin/env python3
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import Yolov8
import rospy

def callback_image(msg):
      global _image
      _image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
def callback_image1(msg):
      global _image1
      _image1 = CvBridge().imgmsg_to_cv2(msg, "bgr8")

if __name__ == "__main__":
    rospy.init_node("try_detect")
    rospy.loginfo("demo node start!")
    image = None
    image_1 = None
    rospy.Subscriber ("/cam2/rgb/image_raw", Image, callback_image)
    rospy.Subscriber("/cam1/rgb/image_raw", Image, callback_image1)
    dnn_yolo = Yolov8("bagv3")
    seat_number = []
    seat=0
    people_number=[]
    people = 0
    y2_number = []
    while not rospy.is_shutdown():
        rospy.Rate (20).sleep()
        if image is None:continue
        image_1 = cv2.flip(image_1,0)
        frame = image.copy()
        boxes = dnn_yolo.forward(image)[0]["det"]
        boxes_1 = dnn_yolo.forward(image_1)[0]["det"]
        for x1, y1, x2, y2, score, class_id in boxes:
            x1, y1, x2, y2, class_id = map(int, [x1, y1, x2, y2, class_id])
            if class_id==58:
                seat_number[seat]=x2-x1
                y2_number[seat]=y2-y1
        for x1, y1, x2, y2, score, class_id in boxes:
            x1, y1, x2, y2, class_id = map(int, [x1, y1, x2, y2, class_id])
            if class_id == 0:
                people_number[people]=x2-x1
        for i in seat:
            for j in people:
                if abs(seat-people)<5:
                    seat_number.pop(i)
                    people_number.pop(j)
                    y2_number.pop(i)
                    j-=1
                    i-=1
                    break
    cv2. imshow ("image", frame)
