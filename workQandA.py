#!/usr/bin/env python3
import rospy
from mr_voice.msg import Voice
import cv2
import yaml


def callback_voice(msg):
    global _voice
    _voice = msg


def text_to_cmd(s):
    global _cmds 
    if _cmds is None or s is None: return None
    s = s.lower()
    for c in _cmds["QnA"]:
        OK = True 
        for i in c["K"]:
            tmp = False
            for j in i:
                if str(j).lower() in s:
                    tmp = True
                    break
            if not tmp: OK = False
        if OK: return c['A']
    return None


if __name__  == "__main__":
    rospy.init_node("demo3")
    rospy.loginfo("demo3 start!")

    _cmds = None
    image = None
    #rospy.Subscriber (" /camera/rgb/image raw", Image, callback_image)
    #dnn yolo = Yolov8 ("yolov8n")
    #seat_number = []
    #seat=0
    #people_number=[]
    #people = 0
    #y2_number = []
    with open("/home/pcms/catkin_ws/src/beginner_tutorials/src/QandA.txt", "r") as f:
        try:
            _cmds = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
    
    _voice = None
    _topic_voice = "/voice/text"
    rospy.Subscriber(_topic_voice, Voice, callback_voice)
'''
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()

        if _voice is not None:
            rospy.loginfo("%s (%d)" % (_voice.text, _voice.direction))
            
            _cmd = text_to_cmd(_voice.text)
            if _cmd is not None:
                cms += ",who favourite drinks is"
        if image is None: continue
        frame = image. copy ()
        boxes = dnn yolo.forward ( image) [0]["det"]
        for x1, y1, x2, 2, score, class_id in boxes:
            x1, y1, x2, y2, class_id = map(int, [x1, y1, x2, y2, class_id])
            if class_id==58:
                seat_number[seat]=x2-x1
                y2_number[seat]=y2-y1
            if class_id == 0:
                people_number[people]=x2-x1
        for i in seat:
            for j in people:
                if abs|seat-people|<5:
                    seat_number.pop(i)
                    people_number.pop(j)
                    y2_number.pop(i)
                    j-=1
                    i-=1
            _cmd += text_to_cmd(_voice.text)
            _voice = None
'''           
    #rospy.loginfo("demo3 end!")

