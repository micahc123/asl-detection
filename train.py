from ultralytics import YOLO

model = YOLO("yolo11n.pt") 

model.train(
    data=f"sign_recognition-2/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name='sign_language_yolo11',
    device='mps'
)
