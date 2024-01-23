import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

if __name__ == "__main__":
    rospy.init_node("test_william")
    print("start")
