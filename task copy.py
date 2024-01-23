#!/usr/bin/env python3.8
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu

from cv_bridge import CvBridge
import cv2
import numpy as np

from std_msgs.msg import String

from mr_voice.msg import Voice

from pcms.openvino_models import HumanPoseEstimation, FaceDetection, PersonAttributesRecognition, Yolov8

from geometry_msgs.msg import Twist
from RobotChassis import RobotChassis
#from tf.transformations import euler_from_quaternion

#from pcms.pytorch_models import *
def callback_character(msg):
    global character
    character = msg.data

def callBack_image(msg):
    global _image
    _image = CvBridge().imgmsg_to_cv2(msg, "bgr8")

def say(text):
    global _pub_speaker
    if text is None: return
    if len(text) == 0: return
    rospy.loginfo("ROBOT: %s" % text)
    _pub_speaker.publish(text)
    rospy.sleep(1)

def callback_depth(msg):
    global depth
    depth = CvBridge().imgmsg_to_cv2(msg, "passthrough")

def callback_voice(msg):
    global _voice
    _voice = msg

def get_real_xyz(x, y):
    global depth
    if depth is None:
        return -1,-1
    h, w = depth.shape[:2]
    d = depth[y][x]
    a = 49.5 * np.pi / 180
    b = 60.0 * np.pi / 180
    real_y = (h / 2 - y) * 2 * d * np.tan(a / 2) / h
    real_x = (w / 2 - x) * 2 * d * np.tan(b / 2) / w
    return real_x, real_y, d

def get_target_d(frame):
    poses = dnn_human_pose.forward(frame)
    frame = dnn_human_pose.draw_poses(frame, poses, 0.1)
    global _image
    image = _image.copy()
    x1, y1, x2, y2 = 1000, 1000, 0, 0
    nlist = []
    dlist = []
    targetd =10000
    if len(poses) != 0:
        for i in range(len(poses)):
            x, y, c = map(int, poses[i][0])
            nlist.append([x,y,i])
        for i in range(len(nlist)):
            _, _, d = get_real_xyz(nlist[i][0], nlist[i][1])
            dlist.append([d,nlist[i][2]])
        for i in range(len(dlist)):
            if dlist[i][0] < targetd:
                targetd = dlist[i][0]
                targeti = dlist[i][1]                
        pose = poses[targeti]
        for i, p in enumerate(pose):
                x, y, c = map(int, p)
                if x < x1 and x != 0: x1 = x
                if x > x2: x2 = x
                if y < y1 and y != 0: y1 = y + 5
                if y > y2: y2 = y
            # cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        dnn_human_pose.forward(frame)
        appearance = dnn_appearance.forward(image)
       
                    
        # rospy.loginfo(appearance)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        # rospy.loginfo(cx,cy)
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
        # return cx, image
        _, _, d = get_real_xyz(cx, cy)
        a = 0
        if d != 0:
            a = max(int(50 - (abs(d - 1150) * 0.0065)),20)
        rospy.loginfo(a)
        print("d : "+str(d))
        cv2.rectangle(image, (x1, y1 - a), (x2, y2), (255, 0, 0), 2)
        if d == -1:
            return -1, -1, -1
        return d, image, appearance
    return -1, -1, -1

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

def detect_face(frame):
    vec_dicts = {}
    global vec_names
    #for name in vec_names:
     #   vec_dicts[name] = np.loadtxt("/home/pcms/catkin_ws/src/beginner_tutorials/src/%s" % name)
    #print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvec_name : "+str(len(vec_names)))
    faces = dnn_face.forward(frame)
    who = -1
    global c
    f = -1
    global cnt
    if len(faces) == 0:return -1, -1
    for face in faces:
        x1, y1, x2, y2 = face
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        d =getDepth(cx,cy)
        who = True
        print("d : ",str(d))
        if d is not None:
            if f == -1 and d < 2900:
                f = [x1,x2,y1,y2]
            if f != -1:
                if cnt == 0:
                    if x1 < f[0] and d < 2900:
                        f = face
                elif cnt == 1:
                    if x1 > f[0] and d < 2900:
                        f = face
                elif cnt == 2:
                    rlist = []
                    for face in faces:
                        if len(rlist) == 0:
                            rlist.append(face)
                        else:
                            flag = False
                            for i in rlist:
                                if face[0] < i[0]:
                                    flag = True
                            if flag == False:
                                c+=1
                                rlist.append(face)     
                    if c == 2:
                        return who, [x1, x2, y1, y2] 
    return who, f
       
def angular_PID(cx, tx):
    e = tx - cx
    p = 0.0015
    z = p * e
    if z > 0:
        z = min(z, 0.25)
        z = max(z, 0.05)
    if z < 0:
        z = max(z, -0.25)
        z = min(z, -0.05)
    return z


def linear_PID(cd, td):
    e = cd - td
    p = 0.0002
    x = p * e
    if x > 0:
        x = min(x, 0.16)
        x = max(x, 0.1)
    if x < 0:
        x = max(x, -0.16)
        x = min(x, -0.1)
    return x
        
def d_set(nlist):
    alist = []
    for i in nlist:
        if not i in alist:
            alist.append(i)
    return alist

def find_Char(alist, guest,gender):
    if not "man" in alist:
        return "is a " + gender, ["man","woman"]
    elif not "woman" in alist:
        return "is a " + gender,["man","woman"]
    else:
        if not "hair" in alist:
            if gender == "man":
                return "has short hair", "hair"
            else:
                return "has long hair", "hair"
        else:                                
            for k,v in guest.items():
                if v == True and k != "is_male" and k != "has_longhair":
                    print("add")
                    return " ".join(k.split("_")), k
                    break
    '''
    elif not "age" in alist:
            if guestAge >= 3 and guestAge <= 11:
                return "is in childhood", "age"
            elif guestAge > 12 and guestAge <= 35:
                return "is in young age", "age"
            elif guestAge > 35 and guestAge <= 64:
                return "is in middle age", "age"
            elif guestAge > 64 and guestAge <= 100:
                return "is in elderly", "age"
    '''
    return "is a " + gender,["man","woman"]

def move_status():
    while not rospy.is_shutdown():
        code = chassis.status_code
        text = chassis.status_text

        if code == 0:       # No plan.
            pass
        elif code == 1:     # Processing.
            pass
        elif code == 3:     # Reach point.
            say("I am arrived.")
            _status = 2
            break
        elif code == 4:     # No solution.
            say("I am trying to move again.")
            break
        else:
            rospy.loginfo("%d, %s" % (code, text))
            break
    
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
        #print(cx,cy)
        d = getDepth(cx,cy)
        #print("d : ",str(d))
        #print(f"x1:{x1},y1:{y1},x2:{x2},y2:{y2}")
        if (x1 != 0 and x2 != 0 and y1 != 0 and y2 != 0) and (x1 < x2 and y1<y2) and (x1 > 0 and x2 >0 and y1 > 0 and y2 > 0): 
            #print(x1,x2,y1,y2)
            x1-=int(d*0.01)
            x2+=int(d*0.01)
            cv2.circle(image,(cx,cy),5,(0,0,255),-1)
            cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),2)
            #print(f"x1:{x1},y1:{y1},x2:{x2},y2:{y2}")
            frame = image[y1:y2,x1:x2,:]
            #cv2.imshow("up",frame)
            Upcolor = detect_color(frame)
            print("upC :",str(Upcolor))
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
                #print(f"x1:{x1},y1:{y1},x2:{x2},y2:{y2}")
                frame = image[y1:y2,x1:x2,:]
                cv2.imshow("down",frame)
                dncolor = detect_color(frame)
                print("dpwnC :",str(dncolor))
                cv2.imwrite(image,"/home/pcms/Desktop/detectColor.png")
                return Upcolor,dncolor
    return -1,-1

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

def getMask(img):
    if image is None:return -1
    h,w,c = image.shape
    if h == 0 or w == 0 or c == 0:
        return -1,-1
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    upperb=np.array([54, 255, 255])
    lowerb=np.array([0, 20, 0])
    mask=cv2.inRange(img, lowerb=lowerb, upperb=upperb)
    masked=cv2.bitwise_and(img, img, mask=mask)
    return (mask, masked)

def check_mask(face_img) -> bool:
    face_img = face_img[int(face_img.shape[0]/2):].copy()
    # check if it wearing mask
    face_img = cv2.resize(face_img, (224, 112))
    mask, masked = getMask(face_img)
    # get total mask pixel
    if mask == -1: return -1
    tot_pixel = np.sum(mask == 255)
    return tot_pixel < 9500

if __name__ == "__main__":
    rospy.init_node("task")
    rospy.loginfo("started task")

    _voice = None
    rospy.Subscriber("/voice/text", Voice, callback_voice)

    _image = None
    rospy.Subscriber("/cam1/rgb/image_raw", Image, callBack_image)
    rospy.wait_for_message("/cam1/rgb/image_raw", Image)

    imagePath = rospy.Publisher("/pcms/imageP", String, queue_size=10)    
    
    character = None
    rospy.Subscriber("/pcms/appearance",String, callback_character)
    
    depth = None
    rospy.Subscriber("/cam1/depth/image_raw", Image, callback_depth)
    
    print("loading dnn")
    path_openvino = "/home/pcms/models/openvino/"
    dnn_face = FaceDetection(path_openvino)
    dnn_appearance = PersonAttributesRecognition(path_openvino)
    dnn_human_pose = HumanPoseEstimation()
    #ddn_rcnn = FasterRCNN()
    #dnn_yolo = Yolov5()
    print("wait for opening yolo")
    dnn_yolo_glasses = Yolov8("v1best0")
    _pub_speaker = rospy.Publisher("/speaker/say", String, queue_size=10)
    
    #imu = None
    #rospy.Subscriber("/imu/data", Imu, imu_callback)
    #rospy.wait_for_message("/imu/data", Imu)
    msg_cmd = Twist()
    pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    print("readys")

    vec_names = []
    vec_dicts = {}
    tlist = []#appearance
    alist = [] # again appearance
    Llist = []#location
    status = 0

    '''
    -1: go to the room with guests in it
    '''

    now_person = ""
    t0 = None
    chassis = RobotChassis()
    cnt = 1
    guest_color = None
    pos = {"waitpoint1_come": (-0.402, -1.58, -1), "waitpoint2_come": (-0.442, -2.86, -1),"waitpoint1_back": (-0.454, -2.87, 0.2),"waitpoint2_back": (-0.153, -1.31, 0.2),"roomL": (-0.422, -3.18, 0),"roomR": (-0.422, -3.18, 2.3), "door":(-0.639,-2.37,0.173), "master2":(0.112, -0.122, 0.188)}
    save = False #status =1
    c = 0
    publish = False

    while not rospy.is_shutdown():
        rospy.Rate(30).sleep()
        msg_cmd.linear.x = 0.0
        msg_cmd.angular.z = 0.0
        tlist = []
        # cv2.imshow("_image",_image)
        frame = _image.copy()
        if status == -1:  # go to the room with guests in it
            print(cnt)
            if _voice == None: continue
            if "dining" in _voice.text and "room" in _voice.text:
                rospy.loginfo(_voice.text)
                #chassis.move_to(*pos["door"])
                chassis.move_to(*pos["waitpoint1_come"])
                chassis.move_to(*pos["waitpoint2_come"])
                if cnt != 1:
                    chassis.move_to(*pos["roomL"])
                else:
                    chassis.move_to(*pos["roomR"])
                move_status()
                #say('I arrived dining room')
                _voice = None
                status += 1
                print("I arrived outside the room")

        if status == 0:
            if cnt != 1:
                msg_cmd.angular.z = -0.2    
            else:
                msg_cmd.angular.z = 0.2
            who, xlist = detect_face(frame)
            if xlist== -1: continue
            x1, y1, x2, y2 = xlist
            print(x1,x2,y1,y2)
            face_img = frame[y1:y2,x1:x2,:]
            cv2.imshow("face_img",face_img)
            
            max_conf = -1
            have_glasses = 0
            try:
                result = dnn_yolo_glasses.forward(face_img)[0]
                for box in result["det"]:
                    if box[4] > max_conf:
                        have_glasses = box[5]
                        print("have_glasses:",str(have_glasses))
                have_glasses = bool(have_glasses)
                print("bool:",str(have_glasses))
                print(f"xlist: {xlist}")
            except Exception:
                pass
            if who != -1 and xlist != -1: 
                cx = ((xlist[0] + xlist[1]) // 2)
                if max(cx, 315) == cx and min(cx, 325) == cx:
                    #status += 1
                    print("finish")
                    msg_cmd.angular.z = 0
                    save = False
                    yolo = False
                    publish = False
                    rospy.sleep(0.05)
                    break
                else:
                    v = angular_PID(cx, 320)
                    msg_cmd.angular.z = v
                    print(v)
            pub_cmd.publish(msg_cmd)
            h,w,c = face_img.shape
            if h != 0 and w != 0 and c != 0:
                cv2.imshow("face_img",face_img)
        if status == 1:
            if frame is None: continue
            d, image, appearance= get_target_d(frame)
            #guest_color = get_target_body_color(frame)
            _frame = _image.copy()
            #if yolo == False:
                #yolo(_frame)
            print(appearance)
            cv2.imshow("test",image)    
            upcolor,dncolor = detect_cloth_color(frame)
            hasMask = check_mask(image)
            
            if publish == False: 
                guest = appearance
                p = "/home/pcms/Desktop/test2.png"
                cv2.imwrite(p,_image)
                imagePath.publish("/home/pcms/Desktop/test2.png")
                print("publish")
                publish = True
            if publish == True and upcolor != -1 and dncolor != -1 and hasMask != -1:
                status +=1
                
                
        if status == 2:
            d, image, appearance = get_target_d(frame)
            if d != -1:
                if d != 0:
                    rospy.loginfo(d)
                    if d < 925 or d > 975:
                        v = linear_PID(d, 950)
                        msg_cmd.linear.x = v
                        print(v)
                    else:
                        say('I found a guest')
                        status += 1
                        print("done")
                        rospy.sleep(0.05)
                        say("What is your name")    
                frame = image
            pub_cmd.publish(msg_cmd)

        if status == -2:
            who, xlist = detect_face(frame)
            if who != -1:
                x1, x2, y1, y2 = xlist
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #cv2.putText(frame, who, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("frame", frame)
                
                cv2.waitKey(1)
                #if who == "Unknown":
                #    _pub_speaker.publish("What is your name")  
                #    cv2.waitKey(1)
                #    status += 1

        if status == 3:
            if _voice is None: continue
            Klist = ["Candy","Kenny"]
            Ilist = ["Hannah","Tina","ivana","Ivana"]
            Wlist = ["William","Lily"]
            if "name" in _voice.text:
                rospy.loginfo(_voice.text)
                v = _voice.text.split(" ")[-1]
                if v == "is" or v == "name" or v not in Klist and v not in Ilist and v not in Wlist:
                    say("Could you repeat your name")
                    _voice = None
                    continue
                elif v in Ilist:
                    now_person = "Ivana"
                elif v in Klist:
                    now_person = "Kenny"
                elif v in Wlist:
                    now_person = "William"
                rospy.loginfo(now_person)
                #np.savetxt("/home/pcms/catkin_ws/src/beginner_tutorials/src/" + v, vec)
                #vec_names.append(v)
                
                
                
                #rospy.loginfo("saved")
                _voice = None
                status += 1
        if status == 4:
            #chassis.move_to(*pos["door"])
            chassis.move_to(*pos["waitpoint1_back"])
            chassis.move_to(*pos["waitpoint2_back"])
            #chassis.move_to(*pos["master2"])
            print(character)
            while character is None:
                rospy.Rate(20).sleep()    
                continue
            tem = []
            print("character : " + character)
            #imagePath.publish(None)
            #tlist.append(character)
            gender = character.split()[1]
            '''
            if "room" not in character and "wall" not in character and "hallway" not in character and "kitchen" not in character and "standing" not in character:
                if "and" in character and "standing" not in character:
                    character = character.split("and")
                    character[0] = list(d_set(character[0].split()[2:]))
                    character[1] = list(d_set(character[1].split()))
                    tem.append(character[0][-1])
                    tem.append(character[1][-1])
                    if character[0][-1] == character[1][-1]:
                        tlist.append(" ".join(character[0]))
                    else:
                        tlist.append(" ".join(character[0]))
                        tlist.append(" ".join(character[1]))
                else:
                    character = list(d_set(character.split()[2:]))    
                    tem.append(character[-1])
                    tlist.append(" ".join(character))
            else:
                character = list(d_set(character.split()[2:]))
            
            print("tem :",tem)
            print("character2 :",character)
            print("tlist :",tlist)
            if len(tlist) > 2:
                tlist.pop(1)
            print("guest :", guest)
            cnt_repeat = 0    
            print("fining")
            print("len(tlist) :" + str(len(tlist)))
            print("len(alist) :" + str(len(alist)))
            add = []
            for i in alist:
                for j in range(len(tlist)):
                    print(tlist[j], i)
                    if i in tlist[j]:
                        print(j)
                        tls, temls= find_Char(alist,guest,gender)
                        tlist[j] = tls
                        if "list" in str(type(temls)):
                            for ti in temls:
                                add.append(ti) 
                        else:
                            add.append(temls)
            for i in range(len(add)):
                alist.append(add[i])
            print("finish check same")                                  
            while len(tlist) < 2:
                print("adding")
                tls, temls= find_Char(alist,guest,gender)
                tlist.append(tls)
                tem.append(temls)   
                for t in tem:  
                    if "list" in str(type(t)):
                        for ti in t:
                            alist.append(ti) 
                    else:
                        alist.append(t)           
            for t in tem:  
                if "list" in str(type(t)):
                    for ti in t:
                        alist.append(ti) 
                else:
                    alist.append(t)
            print("tlist2 :",tlist)
            print("alist :", alist)
            '''
            tlist =[]
            say(f"I found {now_person} in the room")
            if hasMask and "mask" not in alist and len(tlist) < 2:
                tlist.append(f"{now_person} is wearing a mask")
                alist.append("mask")
            if "upcolor" not in alist and len(tlist) <2:
                tlist.append(f"{now_person} is wearing a {upcolor} cloth")
                alist.append("upcolor")
            if "dncolor" not in alist and len(tlist) <2:
                tlist.append(f"{now_person} is wearing a lower cloth in {dncolor}")
                alist.append("dncolor")
            if "gender" not in alist and len(tlist) < 2:
                tlist.append(f"{now_person} is a {gender}")
            if "hair" not in alist and len(tlist) < 2:
                if gender == "man":
                    tlist.append(f"{now_person} has short hair")
                else:
                    tlist.append(f"{now_person} has long hair")
                alist.append("hair")
            for t in tlist:
                say(t)
                rospy.sleep(0.05)
            cnt += 1
            status = -1
            print("status :",status)
            if cnt == 3:
                break
        cv2.imshow("image", frame)
        cv2.waitKey(1)


