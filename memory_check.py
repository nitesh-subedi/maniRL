from ultralytics import SAM, YOLO
import cv2

# Load a model
model = YOLO("yolo11n.pt")

# Display model information (optional)
model.info()

# Run inference with bboxes prompt
# results = model("ultralytics/assets/zidane.jpg", bboxes=[439, 437, 524, 709])
image_path = "/home/nitesh/.local/share/ov/pkg/isaac-sim-4.0.0/maniRL/images/image_goal.jpg"
image = cv2.imread(image_path)

# Run inference with points prompt
results = model(image_path, save=True)
# print(results[0])
