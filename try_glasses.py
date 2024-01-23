#!/usr/bin/env python3.8
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
from pcms.openvino_models import FaceDetection
def callBack_image(msg):
    global _image
    _image = CvBridge().imgmsg_to_cv2(msg, "bgr8")

def get_face_img(frame):
    faces = dnn_face.forward(frame)
    if len(faces) > 0:
        x1, y1, x2, y2 = faces[0]
        if x1 > x2: x1,x2 = x2,x1
        if y1 > y2: y1,y2 = y2,y1
        if (x1 != 0 and x2 != 0 and y1 != 0 and y2 != 0) and (x1 < x2 and y1 < y2) and (
                x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0):
            h, w, c = frame.shape
            if h != 0 and w != 0 and c != 0:
                face_img = frame[y1:y2, x1:x2, :]
                return face_img
    return None

def detectGlasses(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w, c = hsv_frame.shape
    cnt = 0
    if h != 0 and w != 0 and c != 0:
        for x in range(w):
            for y in range(h):
                if hsv_frame[y, x, 2] <= 46:
                    cnt += 1
    if cnt >= 5:
        return True
    return False

def getGlasses(frame):
    if frame is None: return False
    img = frame.copy()
    
    # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(img)
    x1, y1, x2, y2 = 0, 0, 0, 0
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(image=img, landmark_list=faceLms, connections=mpFaceMesh.FACEMESH_TESSELATION,
                                  connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        for id, lm in enumerate(faceLms.landmark):
            # print(lm)
            ih, iw, ic = img.shape
            if id == 55:
                x1, y1 = int(lm.x * iw), int(lm.y * ih)
            elif id == 412:
                x2, y2 = int(lm.x * iw), int(lm.y * ih)
    if x1 != 0 and y1 != 0 and x2 != 0 and y2 != 0:
        glasses_img = frame[y1:y2, x1:x2, :]
        #glasses_img = cv2.resize(glasses_img, (320, 240))
        if detectGlasses(glasses_img):
            cv2.imshow("glasses_img", glasses_img)
            print("yessssssssssssssss")
            cv2.waitKey(1)
            return True
        return False

if __name__ == "__main__":
    rospy.init_node("try_glasses")
    rospy.loginfo("started task")

    _image = None
    rospy.Subscriber("/camera/color/image_raw", Image, callBack_image)
    rospy.wait_for_message("/camera/color/image_raw", Image)
    
    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
    mp_drawing_styles = mp.solutions.drawing_styles
    
    dnn_face = FaceDetection()

    while not rospy.is_shutdown():
        rospy.Rate(30).sleep()
        
        frame = get_face_img(_image)
        if getGlasses(_image):
            print("YES")
        else:
            print("NO")
        