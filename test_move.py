#!/usr/bin/env python3.8
import rospy
from RobotChassis import RobotChassis
from geometry_msgs.msg import Twist
from std_msgs.msg import String

fablab = {"door":(-6.88,-4.83,-1.7),"seat1":(-6.36,-2.86,-4.89),"seat2":(-5.52,-3.01,-4.89),"master":(-4.69,-2.91,-0.18)}

if __name__ == "__main__":
    rospy.init_node("test_move")
    rospy.loginfo("started!")
    chassis = RobotChassis()
    _pub_speaker = rospy.Publisher("/speaker/say",String,queue_size=10)
    for t in fablab.keys():
        chassis.move_to(*fablab[t])
        _pub_speaker.publish("I arrive the %s" % (t))

    rospy.loginfo("end!")   