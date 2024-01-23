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
    dir_images = os.path.join(dir_data, cfg["mode"], "images")
    dir_labels = os.path.join(dir_data, cfg["mode"], "labels")
    if not os.path.exists(dir_data): os.makedirs(dir_data)
    if not os.path.exists(dir_images): os.makedirs(dir_images)
    if not os.path.exists(dir_labels): os.makedirs(dir_labels)
    
    uri_yaml = os.path.join(dir_data, cfg["dir"]["data"] + ".yaml")
    with open(uri_yaml, "w") as f:
        f.write("path: %s\n" % dir_data)
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        f.write("nc: %d\n" % len(cfg["classes"]))
        f.write("names: %s\n" % cfg["classes"])

    _image = None
    topic_image = "/cam2/rgb/image_raw"
    rospy.Subscriber(topic_image, Image, callback_image)
    rospy.wait_for_message(topic_image, Image)

    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        _image = cv2.flip(_image, 0)
        cv2.imshow("image", _image)
        
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break
        elif key_code in [32]:
            s = datetime.today().strftime('%Y%m%d_%H%M%S')
            #i automatic change the path to chair so please turn to other mode such as bag1 or bag2 if u need to train other things --tavis
            cv2.imwrite(os.path.join(dir_images, s + ".jpg"), _image)
            rospy.loginfo("add %s!" % s)
    cv2.destroyAllWindows()
