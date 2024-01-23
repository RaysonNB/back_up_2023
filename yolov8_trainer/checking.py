#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import yaml
import os 


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

    n_total = len(os.listdir(dir_images))
    for i, fname in enumerate(os.listdir(dir_images)):
        print("%03d/%03d: %s" % (i + 1, n_total, fname))
        fname = fname.split(".")[0]
        fimage = os.path.join(dir_images, fname + ".jpg")
        flabel = os.path.join(dir_labels, fname + ".txt")
        if not os.path.exists(flabel): continue

        frame = cv2.imread(fimage)
        fh, fw, fc = frame.shape
        with open(flabel, "r") as f:
            lines = f.readlines()
        boxes = []
        for line in lines:
            class_id, x, y, w, h = map(float, line.split(" "))
            x1, y1 = int((x - w / 2) * fw), int((y - h / 2) * fh)
            x2, y2 = int((x + w / 2) * fw), int((y + h / 2) * fh)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.imshow("frame", frame)
        key_code = cv2.waitKey(0)
        if key_code in [27, ord('q')]:
            break
    cv2.destroyAllWindows()
