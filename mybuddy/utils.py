import cv2
import torch

class DepthEstimator:
    def __init__(self, model_type="DPT_Large"):
        # Load MiDaS model for depth estimation
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        
        # Move model to GPU if available
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()
        
        # Set up image transformation pipeline
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform

    def remove_nearest_objects(self, img, threshold=0.5):
        # Load and preprocess the image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)

        # Estimate depth
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            depth = prediction.cpu().numpy()

        # Normalize depth map
        depth_min = depth.min()
        depth_max = depth.max()
        depth_normalized = (depth - depth_min) / (depth_max - depth_min)

        # Create a mask for nearest objects
        mask = depth_normalized > threshold
        mask = ~mask  # Invert mask

        # Apply mask to remove nearest objects
        result = img.copy()
        result[mask] = [0, 0, 0]  # Set to white (or any other color)

        return mask.sum(), result

    def get_depth_map(self, image_path):
        # Load and preprocess the image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)

        # Estimate depth
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            depth = prediction.cpu().numpy()

        return depth