import torch
import yaml
import os
import numpy as np
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

    def detect_face(self, image: Image.Image, prompt: str = "face") -> Image.Image:
        self.load_model()
        
        w, h = image.size
        mask_image = Image.new("L", (w, h), 0)
        
        if self.model == "DUMMY" or self.model is None:
             # Fallback
            from PIL import ImageDraw
            draw = ImageDraw.Draw(mask_image)
            draw.ellipse((w//3, h//4, 2*w//3, h//2), fill=255)
            return mask_image

        try:
            # Ultralytics SAM predict
            # It supports point prompts or box prompts.
            # We simulate a "face center" point prompt.
            points = [[w//2, h//3]]
            labels = [1]
            
            print(f"Predicting mask using SAM at point {points}...")
            results = self.model.predict(image, points=points, labels=labels)
            
            if results and len(results) > 0:
                result = results[0]
                if result.masks is not None:
                    # Get the mask (H, W) -> uint8
                    # result.masks.data is tensor
                    mask_tensor = result.masks.data[0].cpu().numpy()
                    mask_uint8 = (mask_tensor * 255).astype(np.uint8)
                    mask_image = Image.fromarray(mask_uint8).resize((w, h))
            
            return mask_image
            
        except Exception as e:
            print(f"Error generating mask: {e}")
            return mask_image
