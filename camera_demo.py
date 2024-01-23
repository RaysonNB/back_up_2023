#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import Yolov8, FaceDetection
import numpy as np

def callback_image(msg):
    global frame
    frame = CvBridge().imgmsg_to_cv2(msg, "bgr8")

def callback_depth(msg):
    global depth
    depth = CvBridge().imgmsg_to_cv2(msg, "passthrough")

def detect_color(image):
    if image is None:return -1
    h,w,c = image.shape
    if h == 0 or w == 0 or c == 0:
        return -1
    print("h,w:",h,w)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv[:, :, 0]], [0], None, [180], [0, 180])
    a = np.reshape(hist, 180)
    color = np.argmax(a)
    print("color :",str(color))
    s = None
    if 0 <= color <= 10 or 156 <= color <= 180:
        s = "red"
    elif 11 <= color < 25:
        s = "orange"
    elif 26 <= color <= 34:
        s = "yellow"
    elif 35 <= color <= 77:
        s = "green"
    elif 78 <= color <= 124:
        s = "blue"
    elif 125 <= color <= 155:
        s = "purple"
    return s

def detect_cloth_color(image):
    poses = dnn_human_pose.forward(image)
    if len(poses) > 0:
        x1 = int(poses[0][6][0])
        y1 = int(poses[0][6][1])
        x2 = int(poses[0][11][0])
        y2 = int(poses[0][11][1])
        cv2.circle(image,(x1,y1),5,(255,0,0),-1)
        cv2.circle(image,(x2,y2),5,(255,0,0),-1)
        cx = (x1 +x2)//2
        cy = (y1+y2) // 2
        print(cx,cy)
        d = getDepth(cx,cy)
        print("d : ",str(d))
        print(f"x1:{x1},y1:{y1},x2:{x2},y2:{y2}")
        if (x1 != 0 and x2 != 0 and y1 != 0 and y2 != 0) and (x1 < x2 and y1<y2) and (x1 > 0 and x2 >0 and y1 > 0 and y2 > 0): 
            print(x1,x2,y1,y2)
            x1-=int(d*0.01)
            x2+=int(d*0.01)
            cv2.circle(image,(cx,cy),5,(0,0,255),-1)
            cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),2)
            print(f"x1:{x1},y1:{y1},x2:{x2},y2:{y2}")
            frame = image[y1:y2,x1:x2,:]
            #cv2.imshow("up",frame)
            Upcolor = detect_color(frame)
            print("upC :",str(color))
            x1 = int(poses[0][12][0])
            y1 = int(poses[0][12][1])
            x2 = int(poses[0][13][0])
            y2 = int(poses[0][13][1])
            cx = (x1 +x2)//2
            cy = (y1+y2) // 2
            d = getDepth(cx,cy)
            if (x1 != 0 and x2 != 0 and y1 != 0 and y2 != 0) and (x1 < x2 and y1<y2) and (x1 > 0 and x2 >0 and y1 > 0 and y2 > 0):
                #print(x1,x2,y1,y2)
                cv2.rectangle(image,(x1-int(d*0.015),y1),(x2+int(d*0.015),y2),(0,255,0),2)
                print(f"x1:{x1},y1:{y1},x2:{x2},y2:{y2}")
                frame = image[y1:y2,x1:x2,:]
                cv2.imshow("down",frame)
                dncolor = detect_color(frame)
                print("dpwnC :",str(color))
                cv2.imwrite(image,"/home/pcms/Desktop/detectColor.png")
                return Upcolor,dncolor
                
    return -1,-1

def bound(x, y, shape):
    x = min(max(x, 0), shape[1])
    y = min(max(y, 0), shape[0])
    return x, y

def getDepth(cx,cy):
    d = depth[cy][cx]
    if d == 0:
        for i in range(1,2,1):
            for x in range(0-i,0+i,i):
                for y in range(0-i,0+i,i):
                    d = depth[y][x]
                    if d!=0:
                        return d
    return d    

if __name__ == "__main__":
    rospy.init_node("ros_tutorial")
    rospy.loginfo("ros_tutorial node start!")
    
    frame = None
    topic_name = "/camera/color/image_raw"
    rospy.Subscriber(topic_name, Image, callback_image)
    rospy.wait_for_message(topic_name, Image)
    

    depth = None
    rospy.Subscriber("/camera/depth/image_raw", Image, callback_depth)

    dnn_face = FaceDetection()
    dnn_yolo_glasses = Yolov8("v1best0")
    dnn_yolo_glasses.classes = ["have glass", "no glass"]
    while not rospy.is_shutdown():
        rospy.Rate(10).sleep()
        if frame is None: continue
        #frame = cv2.flip(frame,0)
        #cv2.imshow("frame", frame)
        faces = dnn_face.forward(frame)
        if len(faces) >0:
            x1,y1,x2,y2 = faces[0]
            cx = (x1+x2)//2
            cy = (y1+y2) //2
            
            cv2.circle(frame,(cx,cy),5,(0,255,0),-1)
            print(getDepth(cx,cy))
        '''
        max_conf = -1
        have_glasses = 0
        faces = dnn_face.forward(frame)
        for x1,y1,x2,y2 in faces:
            #try:
            w, h = x2-x1, y2-y1
            x1, y1 = bound(x1-w/4, y1-h/4,frame.shape[:2])
            x2, y2 = bound(x2+w/4, y2+h/4,frame.shape[:2])
            x1,y1,x2,y2 = map(int, [x1, y1, x2, y2])
            face_img = frame[y1:y2,x1:x2,:]
            h,w,c = face_img.shape
            if (x1 != 0 and x2 != 0 and y1 != 0 and y2 != 0) and (x1 < x2 and y1<y2) and (x1 > 0 and x2 >0 and y1 > 0 and y2 > 0) and (h != 0 and w != 0 and c != 0): 
                print(x1,y1,x2,y2)
                print(h,w,c)
                cv2.imshow("face",face_img)
                #cv2.waitKey(0)
                result = dnn_yolo_glasses.forward(face_img)[0]
                print(result)
                have_glasses = False
                for box in result["det"]:
                    print(box)
                    if box[4] > 0.5:
                        have_glasses = True
                print("have_glasses:",have_glasses)
                print(f"xlist: {x1,x2,y1,y2}")
                if have_glasses:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)'''
                

            #except Exception:
            #    pass
        cv2.imshow("frame",frame)
        key_code = cv2.waitKey(1)
        if key_code in [ord('q'), 27]:
            break
    
    cv2.destroyAllWindows()
    rospy.loginfo("ros_tutorial node end!")
