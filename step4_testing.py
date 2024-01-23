#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import *
import yaml
from yolov8n_openvino_model import *


def callback_image(msg):
    global _image
    _image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
    

if __name__ == "__main__":
    rospy.init_node("test")
    rospy.loginfo("node start!")
    
    _image = None
    topic_image = "/cam2/rgb/image_raw"
    rospy.Subscriber(topic_image, Image, callback_image)
    rospy.wait_for_message(topic_image, Image)
    
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    dir_data = os.path.join(cfg["dir"]["root"], cfg["dir"]["data"])

    ie = Core()
    path = "./runs/detect/custom/weights/best_openvino_model/best.xml"
    # path = "/home/pcms/models/openvino/yolo/yolov8n/yolov8n.xml"
    # cfg["classes"] = []
    # for i in range(80):
        # cfg["classes"].append("%d" % i)
    net = ie.read_model(model=path)
    device_name = "GPU"
    if device_name != "CPU":
        net.reshape({0: [1, 3, 640, 640]})
    model = ie.compile_model(model=net, device_name=device_name)
    
    while not rospy.is_shutdown():
        rospy.Rate(20).sleep()

        image = _image.copy()
        num_outputs = len(model.outputs)
        preprocessed_image = preprocess_image(image)
        input_tensor = image_to_tensor(preprocessed_image)
        result = model([input_tensor])
        boxes = result[model.output(0)]
        masks = None
        if num_outputs > 1:
            masks = result[model.output(1)]
        input_hw = input_tensor.shape[2:]
        detections = postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=image, pred_masks=masks, n_classes=len(cfg["classes"]))
        # {"det": [[x1, y1, x2, y2, score, label_id], ...]}
        image = draw_results(detections[0], image, cfg["classes"])
        cv2.imshow("frame", image)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break
