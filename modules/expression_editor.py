import torch
from diffusers import StableDiffusionXLInpaintPipeline
from PIL import Image
import yaml
import os
import numpy as np

class ExpressionEditor:
    def __init__(self, config_path="config/settings.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.checkpoint_path = self.config["models"]["expression_editor"]["checkpoint_path"]
        self.device = self.config["device"]
        self.pipeline = None
        self.default_denoising = self.config["processing"].get("default_denoising_strength", 0.55)

    def load_model(self, checkpoint_path=None):
        target_path = checkpoint_path if checkpoint_path else self.checkpoint_path
        
        # Check if we are already loaded with the correct model
        # We need to store current loaded path to know if switch is needed
        if hasattr(self, 'current_loaded_path') and self.current_loaded_path == target_path and self.pipeline is not None:
            return

        # If different model loaded, unload first
        if self.pipeline is not None:
             self.unload_model()

        if not os.path.exists(target_path):
             print(f"Warning: SDXL Checkpoint not found at {target_path}. Using standard hf model or failing.")
             # Fallback to standard model for testing if local file missing
             model_id = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
             print(f"Loading fallback model: {model_id}")
             self.pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                 model_id,
                 torch_dtype=torch.float16,
                 variant="fp16",
                 use_safetensors=True
             )
        else:
            print(f"Loading SDXL Inpainting model from {target_path}...")
            self.pipeline = StableDiffusionXLInpaintPipeline.from_single_file(
                target_path,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            )
        
        self.pipeline.to(self.device)
        # Enable memory optimizations
        self.pipeline.enable_model_cpu_offload() 
        self.current_loaded_path = target_path
        print(f"ExpressionEditor model loaded: {target_path}")

    def unload_model(self):
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            torch.cuda.empty_cache()

    def edit_expression(self, image: Image.Image, mask: Image.Image, prompt: str, base_prompt: str = "", strength: float = 0.55, guidance_scale: float = 7.5, num_inference_steps: int = 25) -> Image.Image:
        self.load_model()
        
        full_prompt = f"{base_prompt}, {prompt}, high quality, detailed"
        negative_prompt = "low quality, bad anatomy, bad hands, text, error"
        
        print(f"Editing expression with prompt: {full_prompt}, strength: {strength}")
        
        # Ensure image and mask are same size
        if image.size != mask.size:
            mask = mask.resize(image.size)
            
        original_width, original_height = image.size
        
        # --- High-Res Face Inpainting Strategy ---
        # 1. Calculate Bounding Box of the mask
        mask_arr = np.array(mask)
        # Find indices where mask > 0
        coords = np.argwhere(mask_arr > 0)
        
        if coords.size == 0:
            print("Warning: Empty mask provided. using full image.")
            # Fallback to previous logic if mask empty (though unlikely)
            bbox = (0, 0, original_width, original_height)
        else:
            # coords is (y, x)
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1 # +1 for exclusive
            
            # 2. Add Padding / Context
            # We want a square crop for SDXL usually, or at least some context
            h = y1 - y0
            w = x1 - x0
            
            # Center
            cy = y0 + h // 2
            cx = x0 + w // 2
            
            # Determine size: max dimension * factor (e.g. 1.5x)
            size = int(max(h, w) * 1.5)
            # Ensure divisible by 8 (cleaning)
            
            # Define new bounds
            x_min = max(0, cx - size // 2)
            y_min = max(0, cy - size // 2)
            x_max = min(original_width, cx + size // 2)
            y_max = min(original_height, cy + size // 2)
            
            # Adjust to be square if possible? SDXL handles non-square but square is safe.
            # Let's just crop to this region
            bbox = (x_min, y_min, x_max, y_max)
            
        print(f"Cropping to bbox: {bbox}")
        
        cropped_image = image.crop(bbox)
        cropped_mask = mask.crop(bbox)
        
        # 3. Resize to SDXL native resolution (1024x1024)
        target_size = (1024, 1024)
        input_image_resized = cropped_image.resize(target_size, Image.LANCZOS)
        input_mask_resized = cropped_mask.resize(target_size, Image.NEAREST)
        
        generator = torch.Generator(device=self.device).manual_seed(42)
        
        # 4. Inpaint High Res
        output_crop = self.pipeline(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            image=input_image_resized,
            mask_image=input_mask_resized,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=1024,
            height=1024,
            generator=generator
        ).images[0]
        
        # 5. Composite back
        # Resize output back to crop size
        crop_w = bbox[2] - bbox[0]
        crop_h = bbox[3] - bbox[1]
        output_crop_resized = output_crop.resize((crop_w, crop_h), Image.LANCZOS)
        
        # Create final image
        final_image = image.copy()
        
        # Optimally we should blend the edges, but simple paste might work if mask is good.
        # Let's paste using the mask as alpha to blend only the changed area?
        # But inpainting changes the whole masked area. 
        # Actually, self.pipeline returns the whole image (in this case 1024x1024).
        # We paste it into the original location.
        
        final_image.paste(output_crop_resized, (bbox[0], bbox[1]))
        
        # Optional: Blend using the mask to allow "seamless" borders if the generation drifted color?
        # Typically Inpainting pipeline preserves unmasked area perfectly, so direct paste is fine
        # UNLESS strength < 1.0 where unmasked areas might shift? 
        # Diffusers Inpaint pipeline preserves unmasked pixels by default behavior (blending latents).
        # So explicit paste is correct.
        
        return final_image
