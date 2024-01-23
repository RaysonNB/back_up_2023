#!/usr/bin/env python3
import rospy
from PIL import Image
from clip_interrogator import Config, Interrogator
from std_msgs.msg import String

def callback_path(msg):
    global path
    path = msg.data


if __name__ == "__main__":
    rospy.init_node("find_prompt")
    #ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
    cfg = Config()
    cfg.apply_low_vram_defaults()
    cfg.device = "cpu"
    ci = Interrogator(cfg)
    print(ci.config.device)

    path = None
    rospy.Subscriber("/pcms/imageP", String, callback_path)
    #rospy.wait_for_message("/pcms/imageP", String)
    rospy.sleep(1)

    appearance = rospy.Publisher("/pcms/appearance", String, queue_size=10)
    while True:
        #print(path)
        rospy.Rate(20).sleep()
        if path is None: continue
        s = path
        print(s)
        #s = input()
        image = Image.open(r"%s" % s).convert('RGB')
        print(ci.generate_caption(image))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        appearance.publish(ci.generate_caption(image))
        # print(ci.interrogate(image))
        print("already sent appearance")
        path = None

