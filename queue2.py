#!/usr/bin/env python3
from typing import Tuple, List
import numpy as np
from mr_voice.msg import Voice
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion
import datetime
import time
import math
from sensor_msgs.msg import Imu
class qq(object):
    def __init__(self) -> None:
        self.pre_x, self.pre_z = 0.0, 0.0

    def find_cx_cy(self) -> Tuple[int, int]:
        global up_image, up_depth, net_pose
        l=[5, 12]  # 保留节点 5 和 12
        h, w = up_image.shape[:2]
        pose = None
        poses = net_pose.forward(up_image)

        # 找到最近的人物
        min_distance = float("inf")
        closest_person = None
        reference_x, reference_y = 100, 100  # 更换为你的参考点
        for i, pose in enumerate(poses):
            # 获取人物中心点坐标
            x, y, preds = self.get_pose_target(pose, l[0])
            if preds <= 0:
                continue
            _,_,distance = self.get_real_xyz(up_depth, x, y)
            # 确保最近人物的距离不大于 1800mm
            if distance < min_distance and distance <= 1800:
                min_distance = distance
                closest_person = pose

        if closest_person is None:
            return 0, 0, up_image, "no"

        # 获取最近人物的关键点坐标
        key_points = []
        for j, num in enumerate(l):
            x, y, preds = self.get_pose_target(closest_person, num)
            if preds <= 0:
                continue
            key_points.append((x, y))

        # 计算最近人物关键点的中心点坐标
        cx, cy = np.mean(key_points, axis=0)

        # 在最近的人物周围绘制边界框
        x_min = int(np.min(key_points, axis=0)[0])
        y_min = int(np.min(key_points, axis=0)[1])
        x_max = int(np.max(key_points, axis=0)[0])
        y_max = int(np.max(key_points, axis=0)[1])
        cv2.rectangle(up_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        return int(cx), int(cy), up_image, "yes"

    def get_pose_target(self, pose, num):
        p = []
        for i in [num]:
            if pose[i][2] > 0:
                p.append(pose[i])
                
        if len(p) == 0:
            return -1, -1, -1  # 返回三个值，最后一个值为 -1 表示预测失败
        return int(p[0][0]), int(p[0][1]), 1  # 返回三个值，最后一个值为 1 表示预测成功
    def get_real_xyz(self, depth, x: int, y: int) -> Tuple[float, float, float]:
        if x < 0 or y < 0:
            return 0, 0, 0

        a = 49.5 * np.pi / 180
        b = 60.0 * np.pi / 180
        d = depth[y][x]
        h, w = depth.shape[:2]
        if d == 0:
            for k in range(1, 15, 1):
                if d == 0 and y - k >= 0:
                    for j in range(x - k, x + k, 1):
                        if not (0 <= j < w):
                            continue
                        d = depth[y - k][j]
                        if d > 0:
                            break
                if d == 0 and x + k < w:
                    for i in range(y - k, y + k, 1):
                        if not (0 <= i < h):
                            continue
                        d = depth[i][x + k]
                        if d > 0:
                            break
                if d == 0 and y + k < h:
                    for j in range(x + k, x - k, -1):
                        if not (0 <= j < w):
                            continue
                        d = depth[y + k][j]
                        if d > 0:
                            break
                if d == 0 and x - k >= 0:
                    for i in range(y + k, y - k, -1):
                        if not (0 <= i < h):
                            continue
                        d = depth[i][x - k]
                        if d > 0:
                            break
                if d > 0:
                    break
        x = x - w // 2
        y = y - h // 2
        real_y = y * 2 * d * np.tan(a / 2) / h
        real_x = x * 2 * d * np.tan(b / 2) / w
        return real_x, real_y, d

    def calc_linear_x(self, cd: float, td: float) -> float:
        if cd <= 0:
            return 0
        e = cd - td
        p = 0.0005
        x = p * e
        if x > 0:
            x = min(x, 0.3)
        if x < 0:
            x = max(x, -0.3)
        return x

    def calc_angular_z(self, cx: float, tx: float) -> float:
        if cx < 0:
            return 0
        e = tx - cx
        p = 0.0025
        z = p * e
        if z > 0:
            z = min(z, 0.2)
        if z < 0:
            z = max(z, -0.2)
        return z

    def calc_cmd_vel(self, image, depth) -> Tuple[float, float]:
        image = image.copy()
        depth = depth.copy()

        cx, cy, frame, yn = self.find_cx_cy()
        if yn == "no":
            cur_x, cur_z = 0, 0
            return cur_x, cur_z,frame,"no"

        print(cx, cy)
        _, _, d = self.get_real_xyz(depth, cx, cy)

        cur_x = self.calc_linear_x(d, 750)
        cur_z = self.calc_angular_z(cx, 320)

        dx = cur_x - self.pre_x
        if dx > 0:
            dx = min(dx, 0.3)
        if dx < 0:
            dx = max(dx, -0.3)

        dz = cur_z - self.pre_z
        if dz > 0:
            dz = min(dz, 0.2)
        if dz < 0:
            dz = max(dz, -0.2)

        cur_x = self.pre_x + dx
        cur_z = self.pre_z + dz

        self.pre_x = cur_x
        self.pre_z = cur_z

        return cur_x, cur_z,frame,"yes"
def say(a):
    global publisher_speaker
    publisher_speaker.publish(a)
def get_pose_target(pose, num):
    if pose is None:
        return -1, -1, -1
    p = []
    for i in [num]:
        if pose[i][2] > 0:
            p.append(pose[i])
    
    if len(p) == 0:
        return -1, -1, -1
    return int(p[0][0]), int(p[0][1]), 1

def move(forward_speed: float = 0, turn_speed: float = 0):
    global _cmd_vel
    msg = Twist()
    msg.linear.x = forward_speed
    msg.angular.z = turn_speed
    _cmd_vel.publish(msg)
def get_real_xyz(dp, x, y):
    a1=49.5
    b1=60.0
    a = a1 * np.pi / 180
    b = b1 * np.pi / 180
    d = dp[y][x]
    h, w = dp.shape[:2]
    if d == 0:
        for k in range(1, 15, 1):
            if d == 0 and y - k >= 0:
                for j in range(x - k, x + k, 1):
                    if not (0 <= j < w):
                        continue
                    d = dp[y - k][j]
                    if d > 0:
                        break
            if d == 0 and x + k < w:
                for i in range(y - k, y + k, 1):
                    if not (0 <= i < h):
                        continue
                    d = dp[i][x + k]
                    if d > 0:
                        break
            if d == 0 and y + k < h:
                for j in range(x + k, x - k, -1):
                    if not (0 <= j < w):
                        continue
                    d = dp[y + k][j]
                    if d > 0:
                        break
            if d == 0 and x - k >= 0:
                for i in range(y + k, y - k, -1):
                    if not (0 <= i < h):
                        continue
                    d = dp[i][x - k]
                    if d > 0:
                        break
            if d > 0:
                break

    x = int(x) - int(w // 2)
    y = int(y) - int(h // 2)
    real_y = round(y * 2 * d * np.tan(a / 2) / h)
    real_x = round(x * 2 * d * np.tan(b / 2) / w)
    return real_x, real_y, d
def turn_to(angle: float, speed: float):
    global _imu
    max_speed = 0.2
    limit_time = 3
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
def callback_imu(msg):
    global _imu
    _imu = msg
if __name__ == "__main__":
    import rospy
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    import cv2
    from pcms.openvino_models import HumanPoseEstimation, Yolov8
    import numpy as np
    from geometry_msgs.msg import Twist


    def callback_image(msg):
        global _image
        _image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
        

    def callback_depth(msg):
        global _depth
        _depth = CvBridge().imgmsg_to_cv2(msg, "passthrough")
    

    rospy.init_node("FollowMe")
    rospy.loginfo("FollowMe started!")
    
    # RGB Image Subscriber
    _image = None
    _topic_image = "/cam2/rgb/image_raw"
    rospy.Subscriber(_topic_image, Image, callback_image)
    rospy.wait_for_message(_topic_image, Image)
    
    # Depth Image Subscriber
    _depth = None
    _topic_depth = "/cam2/depth/image_raw"
    rospy.Subscriber(_topic_depth, Image, callback_depth)
    rospy.wait_for_message(_topic_depth, Image)
    net_pose = HumanPoseEstimation(device_name="GPU")
    # cmd_vel Publisher
    _msg_cmd = Twist()
    _pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    publisher_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)
    #Kinda = np.loadtxt(RosPack().get_path("mr_dnn") + "/Kinda.csv")
    print("load")
    _cmd_vel=_pub_cmd
    print("f")
    # Models
    topic_imu = "/imu/data"
    _imu=None
    rospy.Subscriber(topic_imu, Imu, callback_imu)
    rospy.wait_for_message(topic_imu, Imu)
    _net_pose = HumanPoseEstimation()
    #detections = dnn_yolo.forward(_image)[0]["det"]
    # Functions
    _fw = qq()
    # Main loop
    pree_cx=0
    pree_cy=0
    slocnt=0
    step="1"
    lrc="middle"
    lrcc=0
    angle=0
    needa=0
    peoplecnt=0
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        up_image = _image.copy()
        up_depth = _depth.copy()
        if step=="1":
            l=[5, 12]  # 保留节点 5 和 12
            h, w = up_image.shape[:2]
            pose = None
            poses = net_pose.forward(up_image)

            # 找到最近的人物
            min_distance = float("inf")
            closest_person = None
            reference_x, reference_y = 100, 100  # 更换为你的参考点
            for i, pose in enumerate(poses):
                # 获取人物中心点坐标
                x, y, preds = get_pose_target(pose, l[0])
                if preds <= 0:
                    continue
                _,_,distance = get_real_xyz(up_depth, x, y)
                # 确保最近人物的距离不大于 1800mm
                if distance < min_distance and distance <= 1800:
                    min_distance = distance
                    closest_person = pose

            if closest_person is None:
                continue

            # 获取最近人物的关键点坐标
            key_points = []
            for j, num in enumerate(l):
                x, y, preds = get_pose_target(closest_person, num)
                if preds <= 0:
                    continue
                key_points.append((x, y))

            # 计算最近人物关键点的中心点坐标
            cx, cy = np.mean(key_points, axis=0)

            # 在最近的人物周围绘制边界框
            x_min = int(np.min(key_points, axis=0)[0])
            y_min = int(np.min(key_points, axis=0)[1])
            x_max = int(np.max(key_points, axis=0)[0])
            y_max = int(np.max(key_points, axis=0)[1])
            cxg = (x_max - x_min) // 2 + x_min
            cyg = (y_max - y_min) // 2 + y_min
            
            
            x, z, up_image,yn = _fw.calc_cmd_vel(up_image, up_depth)
            
            if yn=="no":
                if slocnt==5:
                    x,z=0,0
                    slocnt=0
                slocnt+=1
            _,docx,doud = get_real_xyz(up_depth, cxg, cyg)
            cv2.circle(up_image, (cxg, cyg), 5, (0, 255, 0), -1)
            print("x:", docx,"d:",doud)
            angle = math.degrees(math.atan(docx/doud))
            if lrcc==0:
                cnt=w//2-cxg
                if cnt<0: lrc="left"
                else: lrc="right"
                lrcc+=1
            step="2"
        if step=="2":
            print(lrc)
            needa=90-angle
            print(needa)
            angle_need=needa//0.2
            print(angle_need)
            for i in range(int(angle_need)):
                if lrc=="left":
                    move(0,-0.2)
                else:
                    move(0,0.2)
                print(i)
                time.sleep(0.02)
            angle_need=docx//0.2
            for i in range(int(angle_need)):
                move(0.2,0)
                time.sleep(0.01)
            angle_need=90//0.2
            for i in range(int(angle_need)):
                if lrc=="left":
                    move(0,0.2)
                else:
                    move(0,-0.2)
                #print(i)
                time.sleep(0.02)
            step="3"
        if step=="3":
            l=[5, 12]  # 保留节点 5 和 12
            h, w = up_image.shape[:2]
            pose = None
            poses = net_pose.forward(up_image)

            # 找到最近的人物
            min_distance = float("inf")
            closest_person = None
            reference_x, reference_y = 100, 100  # 更换为你的参考点
            for i, pose in enumerate(poses):
                # 获取人物中心点坐标
                x, y, preds = get_pose_target(pose, l[0])
                if preds <= 0:
                    continue
                _,_,distance = get_real_xyz(up_depth, x, y)
                # 确保最近人物的距离不大于 1800mm
                if distance < min_distance and distance <= 1800:
                    min_distance = distance
                    peoplecnt=0
                    closest_person = pose

            if closest_person is None:
                peoplecnt+=1
                continue

            # 获取最近人物的关键点坐标
            key_points = []
            for j, num in enumerate(l):
                x, y, preds = get_pose_target(closest_person, num)
                if preds <= 0:
                    continue
                key_points.append((x, y))

            # 计算最近人物关键点的中心点坐标
            cx, cy = np.mean(key_points, axis=0)

            # 在最近的人物周围绘制边界框
            x_min = int(np.min(key_points, axis=0)[0])
            y_min = int(np.min(key_points, axis=0)[1])
            x_max = int(np.max(key_points, axis=0)[0])
            y_max = int(np.max(key_points, axis=0)[1])
            cxg = (x_max - x_min) // 2 + x_min
            cyg = (y_max - y_min) // 2 + y_min
            
            
            x, z, up_image,yn = _fw.calc_cmd_vel(up_image, up_depth)
            
            if yn=="no":
                if slocnt==5:
                    x,z=0,0
                    slocnt=0
                slocnt+=1
            _,_,dout = get_real_xyz(up_depth, cxg, cyg)
            if peoplecnt>=15:
                break
            
                
            _msg_cmd.linear.x = x 
            _msg_cmd.angular.z = z
            _pub_cmd.publish(_msg_cmd) 
        now = datetime.datetime.now()
        filename = now.strftime("%Y-%m-%d_%H-%M-%S.jpg")
        output_dir = "/home/pcms/catkin_ws/src/beginner_tutorials/src/m1_evidence/"
        cv2.imwrite(output_dir + filename, up_image)
        cv2.imshow("frame", up_image)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break
        
    rospy.loginfo("FollowMe end!")

