#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image 
from mr_voice.msg import Voice
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
from RobotChassis import RobotChassis


def callback_image(msg):
    global _image 
    _image = CvBridge().imgmsg_to_cv2(msg, "bgr8")

def callback_depth(msg):
    global _depth 
    _depth = CvBridge().imgmsg_to_cv2(msg, "passthrough")

def callback_voice(msg):
    global _voice
    _voice = msg

def callback_prompt(msg):
    global _prompt
    _prompt = msg


if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("demo start!")

    topic_image = "/cam1/rgb/image_raw"
    topic_depth = "/cam1/depth/image_raw"
    topic_voice = "/voice/text"
    topic_speaker = "/speaker/say"
    topic_cmd = "/cmd_vel"
    topic_prompt = "/pcms/appearance"
    topic_path = "/pcms/imageP"

    _image, _depth = None, None
    rospy.Subscriber(topic_image, Image, callback_image)
    rospy.Subscriber(topic_depth, Image, callback_depth)

    _voice = None
    rospy.Subscriber(topic_voice, Voice, callback_voice)

    pub_speaker = rospy.Publisher(topic_speaker, String, queue_size=10)
    pub_cmd = rospy.Publisher(topic_cmd, Twist, queue_size=10)
    pub_path = rospy.Publisher(topic_path, String, queue_size=10)

    _prompt = None
    rospy.Subscriber(topic_prompt, String, callback_prompt)

    rospy.loginfo("waiting for image...")
    while _image is None: rospy.Rate(20).sleep()
    rospy.loginfo("image received.")

    rospy.loginfo("waiting for depth...")
    while _depth is None: rospy.Rate(20).sleep()
    rospy.loginfo("depth received.")

    # rospy.loginfo("waiting for prompt...")
    # pub_path.publish("/home/pcms/Desktop/test2.png")
    # while _prompt is None: rospy.Rate(20).sleep()
    # rospy.loginfo("prompt received. %s" % _prompt)
    # _prompt = None

    _robot = RobotChassis()

    pub_speaker.publish("I am ready.")
    rospy.sleep(1)

    _status = 0
    _pos = {"master": (-0.0985, -0.37, 0.178), "room": (-0.422, -3.18, 0.154), "door":(-0.639,-2.37,0.173), "master2":(-0.325, -0.147, 0.182)}
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()

        if _status == 0:
            if _voice is None: continue
            if "go" in _voice.text:
                pub_speaker.publish("OK, I am going.")
                _status = 1
        elif _status == 1:
            _robot.move_to(*_pos["room"])
            while not rospy.is_shutdown():
                code = _robot.status_code
                text = _robot.status_text

                if code == 0:       # No plan.
                    pass
                elif code == 1:     # Processing.
                    pass
                elif code == 3:     # Reach point.
                    rospy.loginfo("3. Move to %.2f, %.2f, %.2f" % (P[0], P[1], P[2]))
                    pub_speaker.publish("I am arrived.")
                    _status = 2
                    break
                elif code == 4:     # No solution.
                    rospy.loginfo("To %.2f, %.2f, %.2f" % (G[0], G[1], G[2]))
                    pub_speaker.publish("I am trying to move again.")
                    break
                else:
                    rospy.loginfo("%d, %s" % (code, text))
                    break
        elif _status == 2:
            pass


        frame = _image.copy()
        cv2.imshow("frame", frame)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break

    rospy.loginfo("demo end!")
