#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
from RobotChassis import RobotChassis
from pcms.openvino_models import Yolov8,HumanPoseEstimation
from mr_voice.msg import Voice
from std_msgs.msg import String
import yaml
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion
import numpy as np

task3 = {"start":(11.4,3.2,0.7),"door":(12.5,3.99,-0.2),"seats":(11.1,5.45,3.14)}

with open("/home/pcms/keys.txt", "r") as f:
    try:
        _keys = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(e)

fruit = ['apple','orange']
def callback_image(msg):
    global _frame
    _frame = CvBridge().imgmsg_to_cv2(msg, "bgr8")

def callback_voice(msg):
    global _voice
    _voice = msg

def callback_depth(msg):
    global depth
    depth = CvBridge().imgmsg_to_cv2(msg,"passthrough")

def say(t):
    global _pub_speaker
    if t is None or len(t) == 0:return
    _pub_speaker.publish(t)
    rospy.loginfo(t)
    rospy.sleep(2)
    
def get_depth(cx,cy):
    global depth
    d = depth[cy][cx]
    if d == 0:
        for i in range(-1,2,1):
            for j in range(-1,2,1):
                d = depth[cy+i][cx+j]
                if d != 0:return d
    else:
        return d
    return -1
    
def text_to_ans(s):
    s = s.lower()
    for c in _keys["QnA"]:
        OK = True
        for i in c["K"]:
            tmp = False
            for j in i:
                if str(j).lower() in s:
                    tmp = True
                    break
            if not tmp: OK = False
        if OK: return c["A"]
    return None

def imu_callback(msg: Imu):
        global imu
        imu = msg

def move(forward_speed: float = 0, turn_speed: float = 0):
        global pub_cmd
        msg = Twist()
        msg.linear.x = forward_speed
        msg.angular.z = turn_speed
        pub_cmd.publish(msg)
    
def turn_to(angle: float, speed: float):
    global imu
    max_speed = 1.82
    limit_time = 8
    start_time = rospy.get_time()
    while True:
        q = [
            imu.orientation.x,
            imu.orientation.y,
            imu.orientation.z,
            imu.orientation.w
        ]
        roll, pitch, yaw = euler_from_quaternion(q)
        e = angle - yaw
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

if __name__ == "__main__":
    rospy.init_node("task3")
    rospy.loginfo("task3 started!")
    
    _frame = None
    rospy.Subscriber("/camera/rgb/image_raw",Image,callback_image)
    
    depth = None
    rospy.Subscriber("/camera/depth/image_raw", Image, callback_depth)

    _voice = None
    rospy.Subscriber("/voice/text", Voice, callback_voice)

    _pub_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)

    chassis = RobotChassis()
    
    msg_cmd = Twist()
    pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    dnn_pose = HumanPoseEstimation(device_name="GPU")

    dnn_yolo = Yolov8("yolov8n",device_name="GPU")
    imu = None
    rospy.Subscriber("/imu/data",Imu,imu_callback)
    name = None
    app = None
    step = 1
    

    guest = []

    cnt = 0

    angular = [5,40,70,90,110]
    say("I am ready")
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        if _frame is None:continue
        if imu is None:continue
        if step == 1:
            #detect a guest
            chassis.move_to(*task3["start"])
            h,w,c = _frame.shape
            image = _frame.copy()[:,w//2-20:w,:]
            poses = dnn_pose.forward(image)
            if len(poses) == 0:continue
            say("I found a guest")
            cv2.imshow("img",image)
            
            chassis.move_to(*task3["door"])
            step += 1
        elif step == 2:
            #get name and drink
            say("Welcome to the party")
            say("I am Fambot, what is your name")
            while _voice is None or "name" not in _voice.text:continue
            name = text_to_ans(_voice.text)
            if name is None:
                say("could you please repeat")
                continue
            _voice = None
            say("OK. What is your favorite drink")
            while _voice is None or "drink" not in _voice.text:
                rospy.sleep(0.1)
                continue
            app = text_to_ans(_voice.text)
            while app is None:
                say("could you please repeat")
                _voice = None
                continue
            guest.append([name,app])
            say("OK. Please come with me")
            step += 1
        elif step == 3:
            #go and detect master + introduce the guest + find empty seat
            chassis.move_to(*task3["seats"])

            say("I reach")
            angular_empty = None
            empty_seat = False
            cnt_p = 0 #how many people sitting on the chairs are detected
            for i in range(5):
                turn_to(angular[i]*np.pi/180,0.3)
                say(f"{angular[i]} degree")
                image = _frame.copy()
                have_guest = False
                poses = dnn_pose.forward(image)
                image = dnn_pose.draw_poses(image, poses, 0.1)
                for pose in poses:
                    x1, x2, y1, y2 = None, None, None, None
                    for j, p in enumerate(pose):
                        x, y, c = map(int, p)
                        if x <= 0 or y <= 0: continue
                        if x1 is None or x < x1: x1 = x
                        if x2 is None or x > x2: x2 = x
                        if y1 is None or y < y1: y1 = y
                        if y2 is None or y > y2: y2 = y
                        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
                    if x1 is not None and x2 is not None and y1 is not None and y2 is not None:
                        cx  = (x1+x2)//2
                        cy  = (y1+y2)//2
                        d = get_depth(cx,cy)
                        h,w,c = image.shape
                        if 1/3 * w < cx < 2/3 * w and d < 2000 and d != -1:
                            #cv2.imshow("p",image)
                            #cv2.waitKey(0)
                            say("I see a person in front")
                            have_guest = True
                            cnt_p += 1
                        
                            #cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)
                            break
                
                if have_guest:
                    say("there is a person")
                    say("hello")
                else:
                    say("no person")
                    if not empty_seat:
                        cv2.imwrite(f"/home/pcms/Desktop/empty_seat_{cnt+1}.jpg",image)
                        say(f"{guest[cnt][0]}please have a seat in front of me after I introduce you")
                        empty_seat = True
                        angular_empty = i
                    
                    
                    
                    image = _frame.copy()
                    boxes = dnn_yolo.forward(image)[0]["det"]
                    for box in boxes:
                        x1,y1,x2,y2,class_id,score = map(int,box)
                        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
                    
                cv2.imshow("img",image)
                cv2.waitKey(1)
                if (cnt == 0 and cnt_p == 1) or (cnt == 1 and cnt_p == 2):
                    chassis.move_to(task3["seats"][0],task3["seats"][1],-1.57)
                    say(f"This is {guest[cnt][0]}. {guest[cnt][0]}'s favorite drink is {guest[cnt][1]}")
                    if cnt == 1:
                        say(f"The new guest is {guest[0][0]}. {guest[0][0]} is a male and a teenager. He has black hair and wearing white cloth.  ")
                        rospy.sleep(2)
                    if angular_empty is not None:
                        turn_to(angular[angular_empty]*np.pi/180,0.3)
                    turn_to(angular[i%4+1]*np.pi/180,0.3)
                    cv2.imwrite(f"/home/pcms/Desktop/empty_seat_{cnt+1}.jpg",image)
                    say("Please have a seat")
                    break
                    
                    
                '''
                if cnt == 0 and cnt_p == 1 or cnt == 1 and cnt_p == 2:
                    
                    turn_to(0,0.3)
                    msg.angular.z = -0.3
                    pub_cmd.publish(msg)
                    image = _frame.copy()
                    poses = dnn_pose.forward(image)
                    while len(poses) == 0:
                        image = _frame.copy()
                        poses = dnn_pose.forward(image)
                        continue
                    
                    say("i will now introduce the guest")
            '''
            
            step += 1
        elif step == 4:
            cnt += 1
            if cnt == 2:
                break
            step = 1