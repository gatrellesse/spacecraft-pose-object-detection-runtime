from ultralytics import YOLO
import torch
import os

# Load a model from a YAML configuration file and transfer weights from a pretrained model
model = YOLO(r'Q:/PythonWorkspace/ultralytics-main/ultralytics/cfg/models/v8/yolov8n.yaml').load(r'Q:/PythonWorkspace/ultralytics-main/yolov8n.pt')

# Train the model with specified dataset configuration and training parameters
data_path = r'Q:/PythonWorkspace/ultralytics-main/ultralytics/cfg/datasets/coco128.yaml'
epochs = 15
imgsz = 640
batch_size=16


# Check if dataset file exists
if not os.path.exists(data_path):
    print(f"Dataset file not found: {data_path}")
else:
    try:
        # Train the model
        results = model.train(data=data_path, epochs=epochs, imgsz=imgsz)

        # Save the trained model
        torch.save(model.state_dict(), 'yolov8n_trained.pt')
    except Exception as e:
        print(f"Error occurred during training: {e}")
