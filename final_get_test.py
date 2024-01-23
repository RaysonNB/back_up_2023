#!/usr/bin/env python3
import rospy
from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import Yolov8, HumanPoseEstimation
import numpy as np
from geometry_msgs.msg import Twist
from pcms.pytorch_models import *
from pcms.openvino_yolov8 import *
from ikpy.chain import Chain
from ikpy.link import OriginLink
from open_manipulator_msgs.msg import KinematicsPose
from open_manipulator_msgs.srv import SetKinematicsPose, SetKinematicsPoseRequest

def callback_image(msg):
    global _frame
    _frame = CvBridge().imgmsg_to_cv2(msg, "bgr8")


def callback_depth(msg):
    global _depth
    _depth = CvBridge().imgmsg_to_cv2(msg, "32FC1")


def get_real_xyz(x, y):
    global _depth
    a = 55.0 * np.pi / 180
    b = 86.0 * np.pi / 180
    d = _depth[y][x]
    h, w = _depth.shape[:2]
            
    x = int(x) - int(w // 2)
    y = int(y) - int(h // 2)
    real_y = round(y * 2 * d * np.tan(a / 2) / h, 3)
    real_x = round(x * 2 * d * np.tan(b / 2) / w, 3)
    return real_x, real_y, d


def move_to(x, y, z, t):
    service_name = "/goal_task_space_path_position_only"
    rospy.wait_for_service(service_name)
    
    try:
        service = rospy.ServiceProxy(service_name, SetKinematicsPose)
        
        request = SetKinematicsPoseRequest()
        request.end_effector_name = "gripper"
        request.kinematics_pose.pose.position.x = x
        request.kinematics_pose.pose.position.y = y
        request.kinematics_pose.pose.position.z = z
        request.path_time = t
        
        response = service(request)
        return response
    except Exception as e:
        rospy.loginfo("%s" % e)
        return False


if __name__ == "__main__":
    rospy.init_node("ros_tutorial")
    rospy.loginfo("ros_tutorial node start!")

    xyz_goal = [0.288, 0.0, 0.194]
    _frame = None
    rospy.Subscriber("/cam2/rgb/image_raw", Image, callback_image)

    _depth= None
    rospy.Subscriber("/cam2/depth/image_raw", Image, callback_depth)
    
    dnn_yolo = Yolov8("yolov8n", device_name="GPU")
    t = 3.0
    px, py, pz = 0, 0, 0
    move_to(0.288, 0.0, 0.194, t)
    while not rospy.is_shutdown():

        if _frame is None:
            continue
        if _depth is None:
            continue
        frame = _frame
        depth = _depth
        detections = dnn_yolo.forward(frame)[0]["det"]

        for i, detection in enumerate(detections):
            x1, y1, x2, y2, score, class_id = map(int, detection)
            score = detection[4]
            if class_id == 0:
                continue
            if class_id != 39:
                continue
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            px, py, pz = get_real_xyz(cx, cy)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
        cv2.imshow("frame", frame)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break

    rospy.loginfo("ros_tutorial node end!")
