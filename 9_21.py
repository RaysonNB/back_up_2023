#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import random

mleft = []
mright = []

def draw(event,x,y,flags,param):
    if(event == cv2.EVENT_LBUTTONDBLCLK and mleft == []):
        mleft.append(x)
        mleft.append(y)
    elif(event == cv2.EVENT_LBUTTONDBLCLK and mleft != [] and mright == []):
        mright.append(x)
        mright.append(y)
def get_distance(px,py,pz,ax,ay,az,bx,by,bz):
    A,B,C,p1,p2,p3,qx,qy,qz,distance=0,0,0,0,0,0,0,0,0,0
    A=int(bx)-int(ax)
    B=int(by)-int(ay)
    C=int(bz)-int(az)
    p1=int(A)*int(px)+int(B)*int(py)+int(C)*int(pz)
    p2=int(A)*int(ax)+int(B)*int(ay)+int(C)*int(az)
    p3=int(A)*int(A)+int(B)*int(B)+int(C)*int(C)
    #print("1",p1,p2,p3)
    if (p1-p2)!=0 and p3!=0:
        t=(int(p1)-int(p2))/int(p3)
        qx=int(A)*int(t) + int(ax)
        qy=int(B)*int(t) + int(ay)
        qz=int(C)*int(t) + int(az)
        return int(int(pow(((int(qx)-int(px))**2 +(int(qy)-int(py))**2+(int(qz)-int(pz))**2),0.5)))
    return 0
          
            
def callback_depth(msg):
    global depth
    tmp = CvBridge().imgmsg_to_cv2(msg, "passthrough")
    depth = np.array(tmp, dtype=np.float32)


def dfs(x, y, statu):
    global depth_copy, depth_list, cnt
    if x < 1 or y < 1 or x > len(depth_copy[0]) - 2 or y > len(depth_copy) - 2:
        return
    if depth_copy[y][x] != 0:
        return
    depth_copy[y][x] = statu
    cnt += 1
    if x < 2:
        dfs(x + 1, y, statu)
        return
    if y < 2:
        dfs(x, y + 1, statu)
        return

    bx = False
    by = False
    if abs(abs(depth_list[y + 1][x] - depth_list[y][x]) - abs(depth_list[y][x] - depth_list[y - 1][x])) < 2:
        by = True
        dfs(x, y - 1, statu)
        dfs(x, y + 1, statu)
    if abs(abs(depth_list[y][x + 1] - depth_list[y][x]) - abs(depth_list[y][x] - depth_list[y][x - 1])) < 2:
        bx = True
        dfs(x + 1, y, statu)
        dfs(x - 1, y, statu)
    if not bx and not by:
        return
    return


def callback_image(msg):
    global image
    image = CvBridge().imgmsg_to_cv2(msg, "bgr8")


def add_edge():
    for e in range(len(depth_list)):
        depth_list[e].insert(0, depth_list[e][0])  # 最左
        depth_list[e].insert(-1, depth_list[e][-1])  # 最右
    depth_list.insert(0, depth_list[0])
    depth_list.insert(-1, depth_list[-1])


def change_zero():
    for e in range(1, len(depth_list) - 1, 1):
        error = []
        for f in range(1, len(depth_list[e]) - 1, 1):
            if depth_list[e][f] == 0:
                if depth_list[e - 1][f] or depth_list[e - 1][f - 1] or depth_list[e - 1][f + 1] or depth_list[e + 1][f] or depth_list[e + 1][f - 1] or depth_list[e + 1][f + 1] or depth_list[e][f - 1] or depth_list[e][f + 1]:
                    for i in range(-1, 2, 1):
                        for j in range(-1, 2, 1):
                            if depth_list[e + i][f + j] != 0:
                                error.append(depth_list[e + i][f + j])
                    depth_list[e][f] = sum(error) // len(error)
def callback_voice(msg):
    global s
    s = msg.text
def say(a):
    global publisher_speaker
    publisher_speaker.publish(a)
    
def pose_draw(show):
    cx7,cy7,cx9,cy9,cx5,cy5,l,r=0,0,0,0,0,0,0,0
    s1,s2,s3,s4=0,0,0,0
    global ax,ay,az,bx,by,bz
    #for num in [7,9]: #[7,9] left, [8,10] right
    n1,n2,n3=6,8,10
    cx7, cy7 = get_pose_target(pose,n2)
    
    cx9, cy9 = get_pose_target(pose,n3)
    
    cx5, cy5 = get_pose_target(pose,n1)
    if cx7==-1 and cx9!=-1:
        s1,s2,s3,s4=cx5,cy5,cx9,cy9
        cv2.circle(show, (cx5,cy5), 5, (0, 255, 0), -1)
        ax,ay,az = get_real_xyz(depth2,cx5, cy5)
        _,_,l=get_real_xyz(depth2,cx5,cy5)
        cv2.circle(show, (cx9,cy9), 5, (0, 255, 0), -1)
        bx, by, bz = get_real_xyz(depth2,cx9, cy9)
        _,_,r=get_real_xyz(depth2,cx9,cy9)
    elif cx7 !=-1 and cx9 ==-1:
        s1,s2,s3,s4=cx5,cy5,cx7,cy7
        cv2.circle(show, (cx5,cy5), 5, (0, 255, 0), -1)
        ax, ay, az = get_real_xyz(depth2,cx5, cy5)
        _,_,l=get_real_xyz(depth2,cx5,cy5)
        cv2.circle(show, (cx7,cy7), 5, (0, 255, 0), -1)
        bx,by,bz = get_real_xyz(depth2,cx7, cy7)
        _,_,r=get_real_xyz(depth2,cx7,cy7)
    elif cx7 ==-1 and cx9 == -1:
        pass
        #continue
    else:
        s1,s2,s3,s4=cx7,cy7,cx9,cy9
        cv2.circle(show, (cx7,cy7), 5, (0, 255, 0), -1)
        ax, ay, az = get_real_xyz(depth2,cx7, cy7)
        _,_,l=get_real_xyz(depth2,cx7,cy7)
        cv2.circle(show, (cx9,cy9), 5, (0, 255, 0), -1)
        bx, by, bz = get_real_xyz(depth2,cx9, cy9)
        _,_,r=get_real_xyz(depth2,cx9,cy9)
    
    cv2.putText(show, str(int(l)), (s1 + 5, s2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
    cv2.putText(show, str(int(r)), (s3 + 5, s4 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
    return ax, ay, az, bx, by, bz
def get_pose_target(pose,num):
    p = []
    for i in [num]:
        if pose[i][2] > 0:
            p.append(pose[i])
    
    if len(p) == 0: return -1, -1
    return int(p[0][0]),int(p[0][1])    
def get_real_xyz(dp,x, y):
    a = 55.0 * np.pi / 180
    b = 86.0 * np.pi / 180
    d = dp[y][x]
    h, w = dp.shape[:2]
    x = int(x) - int(w // 2)
    y = int(y) - int(h // 2)
    real_y = round(y * 2 * d * np.tan(a / 2) / h)
    real_x = round(x * 2 * d * np.tan(b / 2) / w)
    return real_x, real_y, d
depth_copy = None
depth_list = []
xylist = []
color = {}
biggest_max = []
if __name__ == "__main__":
    rospy.init_node("BW_getdepth")
    rospy.loginfo("BW_getdepth start!")
    image = None
    depth = None
    rospy.Subscriber("/camera/depth/image_raw", Image, callback_depth)
    rospy.Subscriber("/camera/color/image_raw", Image, callback_image)
    rospy.wait_for_message("/camera/depth/image_raw", Image)
    rospy.wait_for_message("/camera/color/image_raw", Image)
    isackcnt=0
    cimage = image.copy()
    xi = 150
    while True:
        k = cv2.waitKey(1)
        if k == -1:
            cv2.imshow("frame", image)
        else:
            cv2.destroyWindow('frame') 
            break
    depth_img = depth.copy()
    while not rospy.is_shutdown():
        rospy.Rate(50).sleep()
        i = 0
        tem_dlist = []
        tem_xylist = []
        if xi >= 600:
            break
        if isackcnt == 0:
            im2=image.copy()
            isackcnt+=1
        while True:
            h, w = depth_img.shape[:2]
            x, y = xi, h - 50 - i
            i += 10
            if i >= 330:
                depth_list.append(tem_dlist)
                xylist.append(tem_xylist)
                break
            d = depth_img[y][x]
            tem_xylist.append((x, y))
            tem_dlist.append(d)
            rospy.loginfo("%.2f" % d)
            gray = depth_img / np.max(depth_img)
            cv2.circle(im2, (x, y), 2, (0, 0, 255), 2)
            cv2.imshow("frame", im2)
            key_code = cv2.waitKey(1)
            if key_code in [ord('q'), 27]:
                break
        xi += 10
        
    add_edge()
    change_zero()
    
    for e in range(1, len(depth_list) - 1, 1):
        for f in range(1, len(depth_list[e]) - 1, 1):
            print(depth_list[e][f], end=" ")
        print()
        
    depth_copy = [[0 for e in range(len(depth_list[0]))] for f in range(len(depth_list))]
    
    for e in range(len(depth_list)):
        for f in range(len(depth_list[0])):
            depth_copy[e][f] = 0
        
    biggest = 0
    statue = 0
    cnt = 0
    for e in range(1, len(depth_list) - 1, 1):
        for f in range(1, len(depth_list[e]) - 1, 1):
            if depth_copy[e][f] == 0:
                cnt = 0
                statue += 1
                dfs(f, e, statue)
                biggest_max.append(cnt)
    for e in range(len(biggest_max)):
        if biggest_max[biggest] < biggest_max[e]:
            biggest = e
    print(f"the biggest flat is the flat {biggest}")
    
    for e in range(1,len(depth_copy)-1,1):
        for f in range(1,len(depth_copy[0])-1,1):
            error = [depth_copy[e+1][f],depth_copy[e-1][f],depth_copy[e][f+1],depth_copy[e][f-1]]
            check = error[0] == error[1] == error[2] == error[3]
            if depth_copy[e][f] not in error and check:
                depth_copy[e][f] = error[0]
                
    
    for e in range(1, statue + 1, 1):
        color[e] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    for e in range(1, len(depth_list) - 1, 1):
        for f in range(1, len(depth_list[e]) - 1, 1):
            print(depth_copy[e][f], end=' ')
        print()
    s = {}
    '''
    max_key = -1
    max_value = 0
    for e in range(1,len(depth_copy)-1,1):
        for f in range(1, len(depth_copy)-1, 1):
            if depth_copy[e][f] not in s:
                s[depth_copy[e][f]] = 1
            else:
                s[depth_copy[e][f]] += 1
            if s[depth_copy[e][f]] > max_value:
                max_value = s[depth_copy[e][f]]    
                max_key = depth_copy[e][f]
    '''
    image_copy = im2.copy()
    for e in range(1, len(xylist) + 1, 1):
        for f in range(1, len(xylist[0]) + 1, 1):
            circle_color = color[depth_copy[e][f]]
            #if depth_copy[e][f] == max_key:
            cv2.circle(image_copy, xylist[e-1][f-1], 2, circle_color, 2)
    cv2.namedWindow('result')
    cv2.setMouseCallback('result',draw)
    
    while True:
        k = cv2.waitKey(1)
        if mleft != [] and mright != []:
            image_copy = cv2.rectangle(image_copy, (mleft[0],mleft[1]), (mright[0],mright[1]), (0,0,255), 2)
            area = abs(mright[1] - mleft[1]) * abs(mright[0] - mright[0])
            print(mleft, mright)
            for e in range(abs(mleft[0] - 150) // 10 + 1, abs(mright[0] - 150)//10+1):
                for f in range(abs(430 - mright[1])//10+2,abs(430 - mleft[1])//10+2):
                    if depth_copy[e][f] not in s:
                        s[depth_copy[e][f]] = 1
                    else:
                        s[depth_copy[e][f]] += 1
            print(s)
            max_key = 0
            accuracy = max(s.values()) / sum(s.values())
            print(max(s.values()))
            print("the accuracy is %.4f" % accuracy)
            mleft.clear()
            mright.clear()
        cv2.imshow("result", image_copy)
        if k == 27:
            break
    cv2.imwrite("/home/pcms/Desktop/test1.png",image_copy)
    cv2.destroyAllWindows()
    rospy.loginfo("BW_getdepth end!")
