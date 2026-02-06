import torch
from diffusers import StableDiffusionXLInpaintPipeline
from PIL import Image
import yaml
import os

class ExpressionEditor:
    def __init__(self, config_path="config/settings.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.checkpoint_path = self.config["models"]["expression_editor"]["checkpoint_path"]
        self.device = self.config["device"]
        self.pipeline = None
        self.default_denoising = self.config["processing"].get("default_denoising_strength", 0.55)

    def load_model(self):
        if self.pipeline is None:
            if not os.path.exists(self.checkpoint_path):
                 print(f"Warning: SDXL Checkpoint not found at {self.checkpoint_path}. Using standard hf model or failing.")
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
                print(f"Loading SDXL Inpainting model from {self.checkpoint_path}...")
                self.pipeline = StableDiffusionXLInpaintPipeline.from_single_file(
                    self.checkpoint_path,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True
                )
            
            self.pipeline.to(self.device)
            # Enable memory optimizations
            self.pipeline.enable_model_cpu_offload() 
            print("ExpressionEditor model loaded.")

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
        
        # Ensure image and mask are compatible sizes
        if image.size != mask.size:
            mask = mask.resize(image.size)
            
        original_width, original_height = image.size
        
        # SDXL requires dimensions to be divisible by 8 (or 16 better)
        # We round to nearest multiple of 8
        width = (original_width // 8) * 8
        height = (original_height // 8) * 8
        
        # Resize if necessary for the inference (avoiding pipeline auto-resize which might be weird)
        input_image = image.resize((width, height), Image.LANCZOS)
        input_mask = mask.resize((width, height), Image.NEAREST)
            
        generator = torch.Generator(device=self.device).manual_seed(42)
        
        output = self.pipeline(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            mask_image=input_mask,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator
        ).images[0]
        
        # Resize back to original size to avoid any aspect ratio drift if we cropped/rounded
        if output.size != (original_width, original_height):
            output = output.resize((original_width, original_height), Image.LANCZOS)
        
        return output
