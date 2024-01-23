#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import random
import time

mleft = []
mright = []

def draw(event,x,y,flags,param):
    if(event == cv2.EVENT_LBUTTONDBLCLK and mleft == []):
        mleft.append(x)
        mleft.append(y)
    elif(event == cv2.EVENT_LBUTTONDBLCLK and mleft != [] and mright == []):
        mright.append(x)
        mright.append(y)

          
            
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
    
    errort=3
    if abs(abs(depth_list[y + 1][x] - depth_list[y][x]) - abs(depth_list[y][x] - depth_list[y - 1][x])) < errort:
        by = True
        dfs(x, y - 1, statu)
        dfs(x, y + 1, statu)
    if abs(abs(depth_list[y][x + 1] - depth_list[y][x]) - abs(depth_list[y][x] - depth_list[y][x - 1])) < errort:
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
    xi = 80
    time.sleep(1)
    
    
    while True:
        k = cv2.waitKey(1)
        if k == -1:
            uimage = cv2.resize(image, (640*2,480*2))
            cv2.imshow("frame", uimage)
        else:
            cv2.destroyWindow('frame') 
            break
    speed=600
    cv2.destroyAllWindows() 
    depth_img = depth.copy()
    while not rospy.is_shutdown():
        rospy.Rate(speed).sleep()
        i = 0
        tem_dlist = []
        tem_xylist = []
        if xi >= 550:
            break
        if isackcnt == 0:
            im2=image.copy()
            isackcnt+=1
        for i in range(0,340,10):
            h, w = depth_img.shape[:2]
            x, y = xi, h - 50 - i
            i += 10
            #if i >= 330:
            #    depth_list.append(tem_dlist)
            #    xylist.append(tem_xylist)
            #    break
            d = depth_img[y][x]
            tem_xylist.append((x, y))
            tem_dlist.append(d)
            #rospy.loginfo("%.2f" % d)
            cv2.circle(im2, (x, y), 2, (0, 0, 255), 2)
            #cv2.imshow("EEE", im2)
            #key_code = cv2.waitKey(1)
            #if key_code in [ord('q'), 27]:
            #    break
        depth_list.append(tem_dlist)
        xylist.append(tem_xylist)
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
        image_copy = cv2.resize(image_copy, (640*2,480*2))
        cv2.imshow("result", image_copy)
        if k == 27:
            break
            
    cv2.imwrite("/home/pcms/Desktop/test1.png",image_copy)
    cv2.destroyAllWindows()
    rospy.loginfo("BW_getdepth end!")
