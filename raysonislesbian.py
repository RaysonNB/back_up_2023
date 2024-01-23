#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from pcms.pytorch_models import *
from std_msgs.msg import String
from mr_voice.msg import Voice


def say(a):
    publisher_speaker.publish(a)

def move(forward_speed: float = 0, turn_speed: float = 0):
    global _cmd_vel
    msg = Twist()
    msg.linear.x = forward_speed
    msg.angular.z = turn_speed
    _cmd_vel.publish(msg)


def find_bottle():
    global ddn_rcnn
    global frame
    global boxes
    
    
    boxes = ddn_rcnn.forward(frame)
    if len(boxes) == 0:
        return "nothing"
    for id, index, conf, x1, y1, x2, y2 in boxes:
        name = ddn_rcnn.labels[index]
        if name != "person":
            return "nothing"
        else:  # name=="suitcase" or name=="backpack":
            print("enter print bottle function 99 thing")
            cv2.putText(frame, name, (x1 + 5, y1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cx = (x2 - x1) // 2 + x1
            cy = (y2 - y1) // 2 + y1
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            return (x1, y1), (x2, y2), (cx, cy), name


def callback_voice(msg):
    global s
    s = msg.text

def callback_image(msg):
    global _frame
    _frame = CvBridge().imgmsg_to_cv2(msg, "bgr8")


def callback_depth(msg):
    global _depth
    _depth = CvBridge().imgmsg_to_cv2(msg, "passthrough")


def raysonisgay():
    global _frame, _depth, _cmd_vel, ddn_rcnn, s, cur_s, pre_x, pre_z

    msg = Twist()
    find_b = find_bottle()
    
    print(find_b)

    if "stop" in s or "strong" in s or "Store" in s or "dog" in s:
        say("ok")
        exit()
    elif "go" in s or "start" in s:
        if find_b != "nothing": 
            print(len(find_b))
            p1, p2, c, name = find_b
            cx, cy = c
            rospy.loginfo("%d, %d" % (cx, cy))
            h, w, c = _frame.shape
            e = w // 2 - cx
            print(e)
            if abs(e) > 20:
                print("abs(e) > 20")
                #rospy.loginfo("if")s
                v = 0.0025 * e
                cur_z = v
                dz = cur_z - pre_z
                if dz > 0:
                    dz = min(dz, 0.1)
                if dz < 0:
                    dz = max(dz, -0.1)
                msg.angular.z = pre_z + dz
            else:
                #rospy.loginfo("else")
                d = _depth[cy][cx]
                rospy.loginfo("d: %d" % d)

                if d > 0 and d < 3000:
                    v1 = 0.0001 * d
                    msg.linear.x = v1
                    cur_x = v1
                    
                    dx = cur_x - pre_x
                    if dx > 0:
                        dx = min(dx, 0.05)
                    if dx < 0:
                        dx = max(dx, -0.05)
                    msg.linear.x = pre_x + dx
                else:
                    msg.linear.x = 0.0
        pre_x, pre_z = msg.linear.x, msg.angular.z
        print(pre_x, pre_z)
        _cmd_vel.publish(msg)   
        print("finish rayson is gay function")


if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")

    _frame = None
    rospy.Subscriber("/camera/rgb/image_raw", Image, callback_image)
    

    _depth = None
    rospy.Subscriber("/camera/depth/image_raw", Image, callback_depth)

    _cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

    rospy.Subscriber("/voice/text", Voice, callback_voice)
    publisher_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)
    s = ""
    
    rospy.sleep(1)
    print(s)
    
    cur_s = "1"
    ddn_rcnn = FasterRCNN()
    rospy.sleep(1)
    pre_x, pre_z = 0.0, 0.0
    frame = _frame
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        #if len(s)==0: continue
        if _frame is None:
            print("frame is None")
            continue
        if _depth is None:
            print("depth is None")
            continue
        raysonisgay()
        
        
        
        cv2.imshow("frame", _frame)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break

    rospy.loginfo("demo node end!")

