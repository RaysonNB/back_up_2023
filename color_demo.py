#!/usr/bin/env python3.8
import rospy
from sensor_msgs.msg import Image
#from sensor_msgs.msg import Imu

from cv_bridge import CvBridge
import cv2
import numpy as np

from std_msgs.msg import String

#from mr_voice.msg import Voice

#from pcms.openvino_models import HumanPoseEstimation, FaceDetection

#from geometry_msgs.msg import Twist
#from RobotChassis import RobotChassis
#from tf.transformations import euler_from_quaternion

#from pcms.pytorch_models import *
def callBack_image(msg):
    global frame
    frame = CvBridge().imgmsg_to_cv2(msg, "bgr8")

if __name__ == "__main__":
    rospy.init_node("color_demo")
    rospy.loginfo("started task")
    _voice = None
    #rospy.Subscriber("/voice/text", Voice, callback_voice)
    #dn_human_pose = HumanPoseEstimation()
    print("finish dnn")
    frame = None
    rospy.Subscriber("/cam2/rgb/image_raw", Image, callBack_image)
    rospy.wait_for_message("/cam2/rgb/image_raw", Image)

    
    #imagePath = rospy.Publisher("/pcms/imageP", String, queue_size=10)    
    
    character = None
    #rospy.Subscriber("/pcms/appearance",String, callback_character)
    
    depth = None
    #rospy.Subscriber("/camera/depth/image_raw", Image, callback_depth)
    
    
    path_openvino = "/home/pcms/models/openvino/"
    #dnn_face = FaceDetection(path_openvino)
    
    _pub_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)
    
    #imu = None
    #rospy.Subscriber("/imu/data", Imu, imu_callback)
    #rospy.wait_for_message("/imu/data", Imu)
    #msg_cmd = Twist()
    #pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    print("readys")
    vec_names = []
    vec_dicts = {}
    tlist = []#appearance
    alist = [] # again appearance
    Llist = []#location
    status = -2
    now_person = ""
    t0 = None
    #chassis = RobotChassis()
    cnt = 0
    guest_color = None
    pos = {"waitpoint1_come": (-3.64, -3.5, -1), "waitpoint2_come": (-0.442, -2.86, -1),"waitpoint1_back": (-3.65, -3.5, 0.2),"waitpoint2_back": (-0.153, -1.31, 0.2),"roomL": (-3.38, -3.18, 0),"roomR": (-3.38, -3.18, 2.3), "door":(-0.639,-2.37,0.173), "master2":(-3.75, -1.17, 0.188)}
    save = False #status =1
    c = 0
    publish = False
    while not rospy.is_shutdown():
        rospy.Rate(30).sleep()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        low_red = np.array([161, 155, 84])
        high_red = np.array([179, 255, 255])
        red_mask = cv2.inRange(hsv_frame, low_red, high_red)
        red = cv2.bitwise_and(frame, frame, mask=red_mask)
        
        low_blue = np.array([94, 80, 2])
        high_blue = np.array([126, 255, 255])
        blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
        blue = cv2.bitwise_and(frame, frame, mask=blue_mask)
        
        # Green color
        low_green = np.array([25, 52, 72])
        high_green = np.array([102, 255, 255])
        green_mask = cv2.inRange(hsv_frame, low_green, high_green)
        green = cv2.bitwise_and(frame, frame, mask=green_mask)

        low_orange = np.array([18, 40, 90])
        high_orange = np.array([27, 255, 255])
        orange_mask = cv2.inRange(hsv_frame, low_orange, high_orange)
        orange = cv2.bitwise_and(frame, frame, mask=orange_mask)
        
        # Every color except white
        low = np.array([0,0,0])
        high = np.array([179,100,130])
        mask = cv2.inRange(hsv_frame, low, high)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        lower_white = np.array([0,0,168])
        upper_white = np.array([172,111,255])
        mask = cv2.inRange(hsv_frame, lower_white, upper_white)
        white = cv2.bitwise_and(frame, frame, mask=mask)

        lower_black = np.array([0,0,0])
        upper_black = np.array([180,255,50])
        mask = cv2.inRange(hsv_frame, lower_black, upper_black)
        black = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow("Frame", frame)
        cv2.imshow("Red", red)
        cv2.imshow("black", black)
        #cv2.imshow("Blue", blue)
        #cv2.imshow("Green", green)
        #cv2.imshow("Result", result)
        #cv2.imshow("White",white)
        key = cv2.waitKey(1)
        if key == 27:
            break
        
        
