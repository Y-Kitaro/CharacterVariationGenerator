import torch
import yaml
import os
import numpy as np
import cv2
from PIL import Image
from .utils import resolve_model_path

from ultralytics.models.sam import SAM3SemanticPredictor

class MaskGenerator:
    def __init__(self, config_path="config/settings.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.model_name = self.config["models"]["mask_generator"].get("segmentation_model_path", "sam3.pt")
        self.device = self.config["device"]
        self.model = None

    def load_model(self):
        if self.model is None:
            try:
                # Load SAM Model Path
                seg_path = self.config["models"]["mask_generator"].get("segmentation_model_path")
                target_model_path = resolve_model_path(seg_path, self.model_name, label="SAM model")

                print(f"Loading SAM3SemanticPredictor ({target_model_path})...")
                
                # Configure Overrides as per user snippet (sam_test.py)
                overrides = dict(
                    conf=0.25,
                    task="segment",
                    mode="predict",
                    model=target_model_path,
                    half=True,  # Use FP16 for speed as requested
                    save=False, # Do not save files to disk
                )
                
                self.model = SAM3SemanticPredictor(overrides=overrides)
                print("SAM3SemanticPredictor loaded.")
                
            except Exception as e:
                print(f"Failed to load SAM3SemanticPredictor: {e}")
                self.model = None

    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
        torch.cuda.empty_cache()

    def generate_mask(self, image: Image.Image, prompt_text: str = "face", dilation_factor=0) -> Image.Image:
        self.load_model()
        
        w, h = image.size
        # Create blank mask
        mask_image = Image.new("L", (w, h), 0)
        
        if self.model == "DUMMY" or self.model is None:
             # Fallback
            from PIL import ImageDraw
            draw = ImageDraw.Draw(mask_image)
            draw.ellipse((w//3, h//4, 2*w//3, h//2), fill=255)
            return mask_image
            
        try:
            target_text = "face"
            image_np = np.array(image)
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            print(f"Predicting mask using SAM3SemanticPredictor (Text: '{prompt_text}')...")
            
            # Predict
            self.model.set_image(image_cv) 
            results = self.model(text=[prompt_text])
            
            if results and results[0].masks is not None:
                # Combine all masks
                masks_data = results[0].masks.data # tensor
                
                # Create
                combined_mask = np.zeros((h, w), dtype=np.uint8)
                
                for mask_tensor in masks_data:
                    m_np = mask_tensor.cpu().numpy()
                     # Resize if needed (SAM masks might be different resolution?)
                    if m_np.shape != (h, w):
                         m_np = cv2.resize(m_np, (w, h), interpolation=cv2.INTER_NEAREST)
                         
                    combined_mask = np.maximum(combined_mask, (m_np * 255).astype(np.uint8))
                
                mask_image = Image.fromarray(combined_mask)
            else:
                print("No masks found.")

        except Exception as e:
            print(f"Error generating mask: {e}")
            import traceback
            traceback.print_exc()
        
        # Apply Dilation/Erosion if requested
        if dilation_factor != 0:
            try:
                # Convert PIL to Numpy
                mask_np = np.array(mask_image)
                
                k_size = int(abs(dilation_factor)) * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                
                if dilation_factor > 0:
                    mask_np = cv2.dilate(mask_np, kernel, iterations=1)
                else:
                    mask_np = cv2.erode(mask_np, kernel, iterations=1)
                    
                mask_image = Image.fromarray(mask_np)
            except Exception as e:
                print(f"Error applying mask adjustment: {e}")

        return mask_image
