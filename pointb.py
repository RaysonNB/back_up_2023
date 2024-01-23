#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import HumanPoseEstimation
import numpy as np
from geometry_msgs.msg import Twist
from pcms.pytorch_models import *
import math

def callback_image(msg):
    global _frame
    _frame = CvBridge().imgmsg_to_cv2(msg, "bgr8")
def say(a): 
    publisher_speaker.publish(a) 

def callback_depth(msg):
    global _depth
    _depth = CvBridge().imgmsg_to_cv2(msg, "passthrough")
    
def get_real_xyz(x, y):
    global _depth
    a = 49.5 * np.pi / 180
    b = 60.0 * np.pi / 180
    d = _depth[y][x]
    h, w = _depth.shape[:2]            
    x = int(x) - int(w // 2)
    y = int(y) - int(h // 2)
    real_y = round(y * 2 * d * np.tan(a / 2) / h)
    real_x = round(x * 2 * d * np.tan(b / 2) / w)
    print(d)
    return real_x, real_y, d
    
class ColorDetector(object):
    
    def __init__(self, lower, upper, min_size=1000):
        self.lower = lower
        self.upper = upper
        self.min_size = min_size
        
    def get_mask(self, rgb_image):
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask
    
    def find_contours(self, mask):
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_size:
                results.append(cnt)
        results.sort(key=cv2.contourArea, reverse=True)
        return results
    
    def find_center(self, cnt):
        m = cv2.moments(cnt)
        if m["m00"] != 0:
            x = int(np.round(m["m10"] / m["m00"]))
            y = int(np.round(m["m01"] / m["m00"]))
            return x, y
        return 0, 0

    def physical_distance(self, depth_image, x, y, angle=0, max_range=25):
        radian = float(angle) * math.pi / 180
        
        real_x = 0
        real_y = 0
        real_z = 0
        
        
        h, w = depth_image.shape
        flag = False
        e = 0
        while not flag and e < max_range:
            depth = depth_image[max(cy - e, 0):min(cy + e, h), max(cx - e, 0):min(cx + e, w)].copy()
            indices = np.nonzero(depth)
            if len(indices[0]) > 0:
                real_z = np.min(depth[indices])
                flag = True
            else:
                e = e + 1
        
        FOV_H = 60.0
        d = real_z
        lw = d * math.tan(FOV_H / 2 * math.pi / 180)
        lx = float(x) / w * lw * 2 - w / 2
        real_x = lx
        
        FOV_V = 49.5
        d = real_z
        lh = d * math.tan(FOV_V / 2 * math.pi / 180)
        ly = float(y) / h * lh * 2 - h / 2
        real_y = ly
        
        real_x = real_x
        real_y = real_y + real_z * math.sin(radian)
        real_z = real_z * math.cos(radian)
                
        return real_x, real_y, real_z
    

if __name__ == "__main__": 
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")
    
    _frame = None
    rospy.Subscriber("/cam1/rgb/image_raw", Image, callback_image)

    _depth = None
    rospy.Subscriber("/cam1/depth/image_raw", Image, callback_depth)
    
    _cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    
    #arm = Manipulator()
    #chassis = Kobuki()
    detector1 = ColorDetector((100, 16, 16), (110, 255, 255))
    detector2 = ColorDetector((50, 16, 16), (100, 255, 255))
    mask=0
    
    key = 0
    is_turning = False
    while not rospy.is_shutdown():
        rospy.Rate(10).sleep()
        if _frame is None: continue
        if _depth is None: continue
        
        rgb_image = _frame.copy()
        mask = detector1.get_mask(rgb_image)
        cnts = detector1.find_contours(mask)
        if len(cnts) > 0:
            cv2.drawContours(rgb_image, [cnts[0]], 0, (0, 255, 0), 2)
            cx, cy = detector1.find_center(cnts[0])
            cv2.circle(rgb_image, (cx, cy), 5, (0, 0, 255), -1)
            
            real_x, real_y, d = get_real_xyz(cx,cy)
            
            rx, ry, rz = detector1.physical_distance(_depth, cx, cy, 30)
            rx = rx + 40
            rospy.loginfo("%.1f mm, %.1f mm, %.1f mm %d" % (rx, ry, rz,d))
        mask = detector2.get_mask(rgb_image)
        cnts = detector2.find_contours(mask)
        if len(cnts) > 0:
            cv2.drawContours(rgb_image, [cnts[0]], 0, (0, 255, 0), 2)
            cx, cy = detector2.find_center(cnts[0])
            cv2.circle(rgb_image, (cx, cy), 5, (0, 0, 255), -1)
            
            real_x, real_y, d = get_real_xyz(cx,cy)
            
            rx, ry, rz = detector2.physical_distance(_depth, cx, cy, 30)
            rx = rx + 40
            rospy.loginfo("%.1f mm, %.1f mm, %.1f mm %d" % (rx, ry, rz,d))            
            '''
            cv2.putText(rgb_image, "%.1fcm, %.1fcm, %.1fcm" % (rx/10, ry/10, rz/10),
                       (cx + 20, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2,
                       cv2.LINE_AA
            )
            edew2q1qqmk
            if key == 32 or is_turning:
                is_turning = True

                ay = -(ry / 10 - 36.5)
                az = -rx / 10
                ax = rz / 10 - 18 + 6 * math.cos(math.atan2(rx, rz))# * (-rx/abs(rx))

                if abs(az) < 5:
                    is_turning = False
                    chassis.move(0, 0)
                    arm.exec_servos_pos(7, 10, 0, mode=2)
                    arm.open()
                    arm.wait()
                    arm.exec_servos_pos(ax, ay, az, 0, mode=1)
                    arm.wait()
                    arm.close(25)
                    arm.wait()
                    arm.exec_servos_pos(7, 15, 0)
                    arm.wait()
                    arm.exec_servos_pos(ax, ay, az, -15)
                    arm.wait()
                    arm.open()
                    arm.wait()
                    arm.exec_servos_pos(7, 10, 0, mode=2)
                    arm.wait()
                else:
                    if az > 0:
                        chassis.move(0, 0.3)
                    else:
                        chassis.move(0, -0.3)
                    cv.waitKey(100)
                '''
                
            # cv.imshow("depth", camera.depth_image)
        cv2.imshow("image", rgb_image)
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            break

