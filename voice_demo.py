#!/usr/bin/env python3
import rospy
from std_msgs.msg import String

tlist = ["Tom is in a white and black shirt", "Tom has a long hair"]
if __name__ == "__main__":
    rospy.init_node("ros_tutorial")
    rospy.loginfo("ros_tutorial node start!")
    
    speaker = rospy.Publisher("/speaker/say", String, queue_size=10, latch=True)
    for i in tlist:
        #d=i.split(" ")
        speaker.publish(i)
        #for j in d:
         #   rospy.sleep(0.05)
          #  speaker.publish(j)
        rospy.sleep(1)

    rospy.loginfo("ros_tutorial node end!")
