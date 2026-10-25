from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")
results = model.track("min.mp4", save=True, persist=True)
print("done")