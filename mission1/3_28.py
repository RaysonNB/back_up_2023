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
def get_real_xyz(x, y):
    global _depth1
    a = 49.5 * np.pi / 180
    b = 60.0 * np.pi / 180
    d = _depth1[y][x]
    h, w = _depth1.shape[:2]
    if d == 0:
        for k in range(1, 15, 1):
            if d == 0 and y - k >= 0:
                for j in range(x - k, x + k, 1):
                    if not (0 <= j < w):
                        continue
                    d = _depth1[y - k][j]
                    if d > 0:
                        break
            if d == 0 and x + k < w:
                for i in range(y - k, y + k, 1):
                    if not (0 <= i < h):
                        continue
                    d = _depth1[i][x + k]
                    if d > 0:
                        break
            if d == 0 and y + k < h:
                for j in range(x + k, x - k, -1):
                    if not (0 <= j < w):
                        continue
                    d = _depth1[y + k][j]
                    if d > 0:
                        break
            if d == 0 and x - k >= 0:
                for i in range(y + k, y - k, -1):
                    if not (0 <= i < h):
                        continue
                    d = _depth1[i][x - k]
                    if d > 0:
                        break
            if d > 0:
                break

    x = int(x) - int(w // 2)
    y = int(y) - int(h // 2)
    real_y = round(y * 2 * d * np.tan(a / 2) / h)
    real_x = round(x * 2 * d * np.tan(b / 2) / w)
    return real_x, real_y, d

def pose_draw():
    cx7, cy7, cx9, cy9, cx5, cy5 = 0, 0, 0, 0, 0, 0
    global ax, ay, az, bx, by, bz
    #for num in [7,9]: #[7,9] left, [8,10] right
    n1, n2, n3 = 6, 8, 10
    cx7, cy7 = get_pose_target(pose, n2)

    cx9, cy9 = get_pose_target(pose, n3)

    cx5, cy5 = get_pose_target(pose, n1)
    if cx7 == -1 and cx9 != -1:
        cv2.circle(rgb_image, (cx5, cy5), 5, (255, 0, 0), -1)
        ax, ay, az = get_real_xyz(cx5, cy5)

        cv2.circle(rgb_image, (cx9, cy9), 5, (255, 0, 0), -1)
        bx, by, bz = get_real_xyz(cx9, cy9)
    elif cx7 != -1 and cx9 == -1:

        cv2.circle(rgb_image, (cx5, cy5), 5, (255, 0, 0), -1)
        ax, ay, az = get_real_xyz(cx5, cy5)

        cv2.circle(rgb_image, (cx7, cy7), 5, (255, 0, 0), -1)
        bx, by, bz = get_real_xyz(cx7, cy7)
    elif cx7 == -1 and cx9 == -1:
        print("where is your hand")
        #continue
    else:
        cv2.circle(rgb_image, (cx7, cy7), 5, (255, 0, 0), -1)
        ax, ay, az = get_real_xyz(cx7, cy7)

        cv2.circle(rgb_image, (cx9, cy9), 5, (255, 0, 0), -1)
        bx, by, bz = get_real_xyz(cx9, cy9)



if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    _image = None
    rospy.Subscriber("/camera/color/image_raw", Image, callback_image)

    dnn_yolo = Yolov8("yolov8n",device_name="GPU")
    dnn_yolo.classes = ['obj']
    rospy.sleep(1)
    fps, fps_n = 0, 0
    print("no while")
    while not rospy.is_shutdown():
        t1 = time.time()
        rospy.Rate(20).sleep()
        if _image is None:
            continue
        image = _image.copy()
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
            if score > 0.5:
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
