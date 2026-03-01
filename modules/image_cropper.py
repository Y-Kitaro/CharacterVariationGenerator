import numpy as np
from PIL import ImageDraw

class ImageCropper:
    def __init__(self, mask_generator):
        self.mask_gen = mask_generator

    def get_crop_mask_and_box(self, img, prompt, mask_adj=0, conf=0.25):
        if img is None or not prompt:
            return None, None
        
        print(f"Generating crop bounding box for '{prompt}' with adj {mask_adj}, conf {conf}...")
        mask = self.mask_gen.generate_mask(img, prompt_text=prompt, dilation_factor=mask_adj, conf=conf)
        
        if mask is None:
            return None, None
            
        mask_np = np.array(mask)
        y_indices, x_indices = np.where(mask_np > 0)
        if len(y_indices) == 0:
            print("No valid mask region found for cropping.")
            return mask, None
            
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        
        margin = 32
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(img.width, x_max + margin)
        y_max = min(img.height, y_max + margin)
        
        crop_box = (x_min, y_min, x_max, y_max)
        return mask, crop_box

    def preview_crop_mask(self, img, prompt, mask_adj=0, conf=0.25):
        mask, crop_box = self.get_crop_mask_and_box(img, prompt, mask_adj, conf)
        if mask is None:
            return img
        
        preview_img = mask.convert("RGB")
        if crop_box:
            draw = ImageDraw.Draw(preview_img)
            draw.rectangle(crop_box, outline="red", width=3)
        return preview_img

    def crop_image_by_prompt(self, img, prompt, mask_adj=0, conf=0.25):
        mask, crop_box = self.get_crop_mask_and_box(img, prompt, mask_adj, conf)
        if mask is None or crop_box is None:
            return img
            
        img_rgba = img.convert("RGBA")
        img_rgba.putalpha(mask)
        return img_rgba.crop(crop_box)
