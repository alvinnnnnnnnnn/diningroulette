from ultralytics import YOLO 
import time 

model = YOLO("yolov8m.pt")

training_start_time = time.time()

model.train(data = "data_custom.yaml", imgsz = 640, batch = 8, epochs = 50, workers = 0, device = 0)

training_end_time = time.time()
total_training_time = training_end_time - training_start_time
print(f"Total training time: {total_training_time: .2f} seconds")