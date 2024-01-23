#!/usr/bin/env python3
from ultralytics import YOLO
from pathlib import Path


path = "./Downloads/best(1).pt"
det_model = YOLO(path)
label_map = det_model.model.names
print(label_map)

from pathlib import Path
det_model_path = Path("./Downloads/bottlev2/best.xml")
if not det_model_path.exists():
	det_model.export(format="openvino", dynamic=True, half=False)
