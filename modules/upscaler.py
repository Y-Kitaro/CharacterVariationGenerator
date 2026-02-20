import torch
from PIL import Image
import numpy as np
import yaml
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

class Upscaler:
    def __init__(self, config_path="config/settings.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.model_path = self.config["models"]["upscaler"]["model_path"]
        self.device = self.config["device"]
        self.model = None
        self.upsampler = None

    def load_model(self):
        if self.upsampler is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Upscaler model not found at {self.model_path}. Please download it as per README.")
            
            print(f"Loading Upscaler model from {self.model_path}...")
            # RealESRGAN_x4plus_anime_6B uses RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            
            self.upsampler = RealESRGANer(
                scale=4,
                model_path=self.model_path,
                model=model,
                tile=0,  # Tile size, 0 for no tile
                tile_pad=10,
                pre_pad=0,
                half=True if "cuda" in self.device else False,
                device=self.device,
            )
            print("Upscaler model loaded.")

    def unload_model(self):
        if self.upsampler is not None:
            del self.upsampler
            self.upsampler = None
            torch.cuda.empty_cache()
            print("Upscaler model unloaded.")

    def upscale(self, image: Image.Image, scale_factor: float = 2.0) -> Image.Image:
        self.load_model()
        
        # Convert PIL to cv2 image (numpy)
        img_np = np.array(image)
        
        # RealESRGAN expects BGR if read by cv2, but here we have RGB from PIL?
        # basicsr usually works with RGB if strictly configured, but RealESRGANer wrapper assumes cv2-like input?
        # Let's check RealESRGANer source assumption. usually it takes numpy array.
        # It's safer to pass consistent format.
        
        try:
            output, _ = self.upsampler.enhance(img_np, outscale=4) # Model is fixed x4
            # output is numpy array
            output_img = Image.fromarray(output)
            
            # If user wanted scale_factor other than 4, resize
            if scale_factor != 4:
                target_width = int(image.width * scale_factor)
                target_height = int(image.height * scale_factor)
                output_img = output_img.resize((target_width, target_height), Image.LANCZOS)
                
            return output_img
            
        except RuntimeError as e:
            print(f"Error during upscaling: {e}")
            return image
