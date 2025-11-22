from ultralytics import YOLO

# Load an Open Images Dataset V7 pretrained YOLOv8n model
model = YOLO("yolov8n-oiv7.pt")

# Run prediction
results = model.predict(source="test/1.jpg")

# Start training from the pretrained checkpoint
results = model.train(data="dataset/yolo.yaml", epochs=100, imgsz=640)