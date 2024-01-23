#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import FaceDetection, FaceReidentification
import numpy as np
import os


def callback_image(msg):
    global _image 
    _image = CvBridge().imgmsg_to_cv2(msg, "bgr8")


if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.loginfo("demo node start!")

    topic_image = "/camera/rgb/image_raw"
    _image = None
    rospy.Subscriber(topic_image, Image, callback_image)
    rospy.wait_for_message(topic_image, Image)

    pub_faceid = rospy.Publisher("/pcms/faceid", String, queue_size=10)

    path_openvino = "/home/pcms/models/openvino/"
    dnn_face = FaceDetection(path_openvino)
    dnn_faceid = FaceReidentification(path_openvino)

    path_faceid = "/home/pcms/catkin_ws/src/beginner_tutorials/faceid/"
    vec_dicts = {}
    for filename in os.listdir(path_faceid):
        name = filename.split(".")[0]
        vec_dicts[name] = np.loadtxt("%s/%s" % (path_faceid, filename))

    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()
        frame = _image.copy()

        faces = dnn_face.forward(frame)
        for face in faces:
            x1, y1, x2, y2 = face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            img = frame[y1:y2, x1:x2, :]
            vec = dnn_faceid.forward(img)

            name, value = "unknown", 1.0
            for k, v in vec_dicts.items():
                dist = dnn_faceid.compare(vec, v)
                if dist < value and dist < 0.4:
                    name = k
                    value = dist
            cv2.putText(frame, "%s %.2f" % (name, 1 - value), (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            pub_faceid.publish(name)
            
            if cv2.waitKey(1) in [32]:
                name = input("Input your name: ")
                np.savetxt("%s/%s.txt" % (path_faceid, name), vec)
                print("%s, OK!" % name)
                
        cv2.imshow("frame", frame)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break

    rospy.loginfo("demo node end!")
