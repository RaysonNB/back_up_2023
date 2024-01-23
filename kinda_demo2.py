#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image 
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np 
from pcms.openvino_models import Yolov8, HumanPoseEstimation
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import Imu


def callback_image1(msg):
    global _image1
    _image1 = CvBridge().imgmsg_to_cv2(msg, "bgr8")


def callback_depth1(msg):
    global _depth1
    _depth1 = CvBridge().imgmsg_to_cv2(msg, "passthrough")


def callback_image2(msg):
    global _image2
    _image2 = cv2.flip(CvBridge().imgmsg_to_cv2(msg, "bgr8"), 0)


def callback_depth2(msg):
    global _depth2
    _depth2 = cv2.flip(CvBridge().imgmsg_to_cv2(msg, "passthrough"), 0)


def callback_imu(msg):
    global _imu
    _imu = msg


def get_real_xyz(depth, x, y):
    a = 49.5 * np.pi / 180
    b = 60.0 * np.pi / 180
    d = depth[y][x]
    h, w = depth.shape[:2]
    # if d == 0:
    #     for k in range(1, 15, 1):
    #         if d == 0 and y - k >= 0:
    #             for j in range(x - k, x + k, 1):
    #                 if not (0 <= j < w):
    #                     continue
    #                 d = depth[y - k][j]
    #                 if d > 0:
    #                     break
    #         if d == 0 and x + k < w:
    #             for i in range(y - k, y + k, 1):
    #                 if not (0 <= i < h):
    #                     continue
    #                 d = depth[i][x + k]
    #                 if d > 0:
    #                     break
    #         if d == 0 and y + k < h:
    #             for j in range(x + k, x - k, -1):
    #                 if not (0 <= j < w):
    #                     continue
    #                 d = depth[y + k][j]
    #                 if d > 0:
    #                     break
    #         if d == 0 and x - k >= 0:
    #             for i in range(y + k, y - k, -1):
    #                 if not (0 <= i < h):
    #                     continue
    #                 d = depth[i][x - k]
    #                 if d > 0:
    #                     break
    #         if d > 0:
    #             break

    x = int(x) - int(w // 2)
    y = int(y) - int(h // 2)
    real_y = round(y * 2 * d * np.tan(a / 2) / h)
    real_x = round(x * 2 * d * np.tan(b / 2) / w)
    return real_x, real_y, d


def move(forward_speed: float = 0, turn_speed: float = 0):
    global _pub_cmd
    msg = Twist()
    msg.linear.x = forward_speed
    msg.angular.z = turn_speed
    _pub_cmd.publish(msg)


def turn_to(angle: float, speed: float):
    global _imu
    max_speed = 1.82
    limit_time = 8
    start_time = rospy.get_time()
    while True:
        q = [
            _imu.orientation.x,
            _imu.orientation.z,
            _imu.orientation.y,
            _imu.orientation.w
        ]
        roll, pitch, yaw = euler_from_quaternion(q)
        e = angle - yaw
        print(yaw, e)
        if yaw < 0 and angle > 0:
            cw = np.pi + yaw + np.pi - angle
            aw = -yaw + angle
            if cw < aw:
                e = -cw
        elif yaw > 0 and angle < 0:
            cw = yaw - angle
            aw = np.pi - yaw + np.pi + angle
            if aw < cw:
                e = aw
        if abs(e) < 0.01 or rospy.get_time() - start_time > limit_time:
            break
        move(0.0, max_speed * speed * e)
        rospy.Rate(20).sleep()
    move(0.0, 0.0)


def turn(angle: float):
    global _imu
    q = [
        _imu.orientation.x,
        _imu.orientation.y,
        _imu.orientation.z,
        _imu.orientation.w
    ]
    roll, pitch, yaw = euler_from_quaternion(q)
    target = yaw + angle
    if target > np.pi:
        target = target - np.pi * 2
    elif target < -np.pi:
        target = target + np.pi * 2
    turn_to(target, 0.1)
    

if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("demo start!")

    topic_image1 = "/cam1/rgb/image_raw"
    topic_depth1 = "/cam1/depth/image_raw"
    topic_image2 = "/cam2/rgb/image_raw"
    topic_depth2 = "/cam2/depth/image_raw"
    topic_cmd = "/cmd_vel"
    topic_speaker = "/speaker/say"
    topic_imu = "/imu/data"

    _image1, _depth1, _image2, _depth2 = None, None, None, None
    _imu = None
    rospy.Subscriber(topic_image1, Image, callback_image1)
    rospy.Subscriber(topic_depth1, Image, callback_depth1)
    rospy.Subscriber(topic_image2, Image, callback_image2)
    rospy.Subscriber(topic_depth2, Image, callback_depth2)
    rospy.Subscriber(topic_imu, Imu, callback_imu)
    _pub_cmd = rospy.Publisher(topic_cmd, Twist, queue_size=10)
    _pub_speaker = rospy.Publisher(topic_speaker, String, queue_size=10)

    rospy.loginfo("Loading models...")
    dnn_yolo = Yolov8("bagv3")
    dnn_yolo.classes = ['obj']
    rospy.loginfo("models OK.")
    
    rospy.loginfo("waiting for image...")
    rospy.wait_for_message(topic_image1, Image)
    rospy.wait_for_message(topic_image2, Image)
    rospy.loginfo("image received.")

    rospy.loginfo("waiting for depth...")
    rospy.wait_for_message(topic_depth1, Image)
    rospy.wait_for_message(topic_depth2, Image)
    rospy.loginfo("depth received.")

    _pub_speaker.publish("I am ready.")
    rospy.sleep(1)

    h, w, c = 480, 640, 3
    _status = 0
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()

        image1, image2 = _image1.copy(), _image2.copy()
        detections = dnn_yolo.forward(_image2)[0]["det"]
        target = None
        for i in range(len(detections)):
            x1, y1, x2, y2, p, cid = map(int, detections[i])
            cv2.rectangle(image2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if target is None:
                # x1 + (x2 - x1) // 2 = x1 + x2 // 2 - x1 // 2 = x1
                target = (x2 + x1) // 2, (y2 + y1) // 2

        if _status == 0:
            if target is not None:
                cx, cy = target
                rx, ry, rz = get_real_xyz(_depth2, cx, cy)
                print(rx, ry, rz)
                cv2.circle(image2, (cx, cy), 5, (0, 255, 0), -1)

                angle = np.arctan2(rx, rz)
                print(angle)
                turn_to(90*3.14/180,0.1)
                _status = 1
        elif _status == 1:
            _pub_speaker.publish("OK")
            _status = 2
        elif _status == 2:
            cx, cy = w // 2, h // 2
            for i in range(cy + 1, h):
                if _depth2[cy][cx] == 0 or 0 < _depth2[i][cx] < _depth2[cy][cx]:
                    cy = i 
            d = _depth2[cy][cx]
            if d > 0:
                e = d - 550
                if abs(e) < 0.1:
                    _status = 3
                v = 0.001 * e
                print(d, e, v)
                # move(v, 0)
        elif _status == 3:
            _pub_speaker.publish("OK")
            _status = 4
        elif _status == 4:
            pass


        cv2.line(image2, (w // 2, 0), (w // 2, h-1), (0, 255, 0), 1)
        frame = np.zeros((h, w * 2, c), dtype=np.uint8)
        frame[:, :w, :] = image1
        frame[:, w:, :] = image2

        cv2.imshow("frame", frame)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break

    rospy.loginfo("demo end!")
