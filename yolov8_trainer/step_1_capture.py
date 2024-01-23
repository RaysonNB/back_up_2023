#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import yaml
import os 
from datetime import datetime


def callback_image(msg):
    global _image
    _image = CvBridge().imgmsg_to_cv2(msg, "bgr8")


if __name__ == "__main__":
    rospy.init_node("yolov8_trainer")
    rospy.loginfo("node start!")

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    dir_data = os.path.join(cfg["dir"]["root"], cfg["dir"]["data"])
    dir_train_images = os.path.join(dir_data, "train", "images")
    dir_train_labels = os.path.join(dir_data, "train", "labels")
    dir_valid_images = os.path.join(dir_data, "valid", "images")
    dir_valid_labels = os.path.join(dir_data, "valid", "labels")
    if not os.path.exists(dir_data): os.makedirs(dir_data)
    if not os.path.exists(dir_train_images): os.makedirs(dir_train_images)
    if not os.path.exists(dir_train_labels): os.makedirs(dir_train_labels)
    if not os.path.exists(dir_valid_images): os.makedirs(dir_valid_images)
    if not os.path.exists(dir_valid_labels): os.makedirs(dir_valid_labels)
    
    uri_yaml = os.path.join(dir_data, cfg["dir"]["data"] + ".yaml")
    with open(uri_yaml, "w") as f:
        f.write("path: %s\n" % dir_data)
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        f.write("nc: %d\n" % len(cfg["classes"]))
        f.write("names: %s\n" % cfg["classes"])

    _image = None
    topic_image = "/camera/color/image_raw"
    rospy.Subscriber(topic_image, Image, callback_image)
    rospy.wait_for_message(topic_image, Image)
    mode = 1    # 1 for train, 2 for valid.
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()

        frame = _image.copy()
        cv2.putText(frame, "train" if mode == 1 else "valid", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("image", frame)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break
        elif key_code in [ord('1')]:
            mode = 1
        elif key_code in [ord('2')]:
            mode = 2
        elif key_code in [32]:
            dir_images = dir_train_images if mode == 1 else dir_valid_images
            s = datetime.today().strftime('%Y%m%d_%H%M%S')
            cv2.imwrite(os.path.join(dir_images, s + ".jpg"), _image)
            rospy.loginfo("add %s!" % os.path.join(dir_images, s + ".jpg"))
    cv2.destroyAllWindows()
