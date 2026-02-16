import torch
import yaml
import os
import numpy as np
import cv2
from PIL import Image

class MaskGenerator:
    def __init__(self, config_path="config/settings.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.model_name = self.config["models"]["mask_generator"].get("model_name", "sam2.pt")
        self.device = self.config["device"]
        self.model = None

    def load_model(self):
        if self.model is None:
            try:
                from ultralytics import SAM
                print(f"Loading SAM model ({self.model_name}) via Ultralytics...")
                # Ultralytics handles download if not found locally
                self.model = SAM(self.model_name)
                print("SAM model loaded.")
            except Exception as e:
                print(f"Failed to load Ultralytics SAM: {e}")
                self.model = "DUMMY"

    def unload_model(self):
        # Ultralytics models are usually lightweight wrappers, but we can delete the object
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()

    def generate_mask(self, image: Image.Image, points=None, labels=None, dilation_factor=0) -> Image.Image:
        self.load_model()
        
        w, h = image.size
        mask_image = Image.new("L", (w, h), 0)
        
        if self.model == "DUMMY" or self.model is None:
             # Fallback
            from PIL import ImageDraw
            draw = ImageDraw.Draw(mask_image)
            draw.ellipse((w//3, h//4, 2*w//3, h//2), fill=255)
            # return mask_image # Apply dilation even to dummy? Yes.
            
        else:
            try:
                # Ultralytics SAM predict
                # If points are not provided, default to center-ish (which caused issues before, but we keep as fallback)
                if points is None:
                    points = [[w//2, h//3]]
                    labels = [1]
                
                if labels is None:
                    labels = [1] * len(points)
                
                print(f"Predicting mask using SAM at point {points}...")
                # Ultralytics predict expects 'bboxes' or 'points'
                # For points, it likely expects a list of list if multiple, but check docs. 
                # Usually predict(source, points=[[x,y]], labels=[1])
                results = self.model.predict(image, points=points, labels=labels)
                
                if results and len(results) > 0:
                    result = results[0]
                    if result.masks is not None:
                        # Get the mask (H, W) -> uint8
                        # result.masks.data is tensor
                        # Taking the first mask if multiple are returned (SAM often returns 3)
                        # We might want to select the best one, usually the one with highest score, 
                        # but Ultralytics wrapper might just give best.
                        mask_tensor = result.masks.data[0].cpu().numpy()
                        mask_uint8 = (mask_tensor * 255).astype(np.uint8)
                        mask_image = Image.fromarray(mask_uint8).resize((w, h))
                
            except Exception as e:
                print(f"Error generating mask: {e}")
                # return mask_image
        
        # Apply Dilation/Erosion if requested
        if dilation_factor != 0:
            try:
                # Convert PIL to Numpy
                mask_np = np.array(mask_image)
                
                # Kernel size must be positive odd integer usually, but let's check opencv docs.
                # structuring element size (k, k). k should be roughly 2*factor + 1? 
                # Or just factor size. Let's say factor is pixels.
                k_size = int(abs(dilation_factor)) * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                
                if dilation_factor > 0:
                    # Expand mask
                    mask_np = cv2.dilate(mask_np, kernel, iterations=1)
                else:
                    # Shrink mask
                    mask_np = cv2.erode(mask_np, kernel, iterations=1)
                    
                mask_image = Image.fromarray(mask_np)
            except Exception as e:
                print(f"Error applying mask adjustment: {e}")

        return mask_image
