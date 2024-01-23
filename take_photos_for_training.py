#!/usr/bin/env python3
import cv2
import rospy
count = 1
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def callback_photo(msg):
    global image
    image = CvBridge().imgmsg_to_cv2(msg,"bgr8")
    
if __name__ == "__main__":
    rospy.init_node("take_photos_for_training")
    rospy.loginfo("take_photos_for_traning started!")
    image = None
    rospy.Subscriber("cam2/rgb/image_raw",Image,callback_photo)
    rospy.wait_for_message("cam2/rgb/image_raw",Image)
    my_data_path = "/home/pcms/photo_data/"
    for path in os.listdir(my_data_path):
       if os.path.isfile(os.path.join(my_data_path, path)):
           count += 1

    camera = cv2.VideoCapture(0)
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        if image is None:continue
        _image = image.copy()
        cv2.imshow("image", image)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break
        if key_code in [27, ord(' ')]:
            p = "/home/pcms/photo_data/img%d.jpg" % count
            print(p)
            cv2.imwrite(p, image)
            count += 1
    cv2.destroyAllWindows()
    rospy.loginfo("take_photos_for_training end!")
