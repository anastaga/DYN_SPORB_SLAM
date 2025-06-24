#export_yolov8_480x360_onnx.py
from ultralytics import YOLO

IMG_W, IMG_H = 480, 360


model = YOLO("yolov8n.pt")   # or trained .pt

model.export(
    format="onnx",
    imgsz=(IMG_H, IMG_W),   
    opset=17,              
    dynamic=False,           
    simplify=True,
    half=True,               # FP16 weights 
    nms=True
)
print("Exported yolov8n_480x360.onnx succesfully.")

