#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import yaml
import os 


def save():
    global _class_id, _x1, _y1, _x2, _y2, _status
    global frame, fname
    fh, fw, fc = frame.shape
    with open(os.path.join(dir_labels, fname + ".txt"), "w") as f:
        if _x1 >= 0 and _y1 >= 0 and _x2 >= 0 and _y2 >= 0:
            x, y = (_x1 + _x2) // 2 / fw, (_y1 + _y2) // 2 / fh
            w, h = (_x2 - _x1) / fw, (_y2 - _y1) / fh
            f.write("%d %f %f %f %f\n" % (_class_id, x, y, w, h))
    rospy.loginfo("add %s!" % fname)


def callback_mouse(event, x, y, flags, param):
    global _class_id, _x1, _y1, _x2, _y2, _status, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        _x1, _y1 = x, y
        _x2, _y2 = -1, -1
        _status = 1
    if event == cv2.EVENT_MOUSEMOVE:
        if _status == 1:
            _x2, _y2 = x, y
            image = frame.copy()
            cv2.putText(image, "%d" % _class_id, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
            if _x1 != -1 and _y1 != -1 and _x2 != -1 and _y2 != -1:
                cv2.rectangle(image, (_x1, _y1), (_x2, _y2), (0, 255, 0), 2)
            cv2.imshow("frame", image)
    if event == cv2.EVENT_LBUTTONUP:
        _status = 0
    if event == cv2.EVENT_MBUTTONUP:
        save()


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

    _class_id = 0
    _x1, _y1, _x2, _y2 = -1, -1, -1, -1
    _status = 0
    _train = cfg["mode"] == "train"
    _fname = ""
    cv2.namedWindow("frame")  
    cv2.setMouseCallback("frame", callback_mouse)

    n_total = len(os.listdir(dir_images))
    for i, fname in enumerate(os.listdir(dir_images)):
        print("%03d/%03d: %s" % (i + 1, n_total, fname))
        fname = ".".join(fname.split(".")[:-1])
        print(fname)
        fimage = os.path.join(dir_images, fname + ".jpg")
        flabel = os.path.join(dir_labels, fname + ".txt")
        if os.path.exists(flabel): continue

        frame = cv2.imread(fimage)
        cv2.imshow("frame", frame)
        key_code = cv2.waitKey(0)
        if key_code in [27, ord('q')]:
            break
        elif key_code in [32]:
            save()
    cv2.destroyAllWindows()
