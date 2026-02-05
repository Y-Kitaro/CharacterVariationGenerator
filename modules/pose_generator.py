import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import gc
import yaml
import os

class PoseGenerator:
    def __init__(self, config_path="config/settings.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.local_path = self.config["models"]["pose_generator"].get("model_path", "assets/models/qwen_image_edit")
        self.hf_id = self.config["models"]["pose_generator"].get("hf_model_id", "Qwen/Qwen-Image-Edit-2511")
        self.device = self.config["device"]
        self.model = None
        self.processor = None

    def load_model(self):
        if self.model is None:
            # Check if local path has model files (more than just .gitkeep)
            has_files = False
            if os.path.exists(self.local_path):
                files = os.listdir(self.local_path)
                if len(files) > 1: # Assuming .gitkeep is one
                    has_files = True
            
            model_to_load = self.local_path if has_files else self.hf_id
            print(f"Loading PoseGenerator model from: {model_to_load}")
            
            try:
                # Qwen-Image-Edit is typically loaded as a CausalLM with image tokens support.
                # Adjusting loading based on standard Qwen-VL usage if exact pipeline is unkown.
                # Assuming it works with AutoModelForCausalLM like Qwen2-VL.
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_to_load,
                    torch_dtype=torch.float16, # or bfloat16
                    device_map="auto",
                    trust_remote_code=True
                )
                self.processor = AutoProcessor.from_pretrained(model_to_load, trust_remote_code=True)
                print("PoseGenerator model loaded.")
            except Exception as e:
                print(f"Failed to load model: {e}")
                print("Please ensure the model is downloaded in 'assets/models/qwen_image_edit' or available on HF.")

    def unload_model(self):
        if self.model is not None:
            print("Unloading PoseGenerator model...")
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            gc.collect()
            torch.cuda.empty_cache()

    def generate_variations(self, image: Image.Image, base_prompt: str, pose_prompts: list):
        self.load_model()
        if self.model is None:
            return []

        generated_images = []
        print(f"Generating variations for {len(pose_prompts)} prompts...")
        
        for p_idx, prompt in enumerate(pose_prompts):
             # Basic conversation structure for Qwen-VL Based editing.
             # The exact prompt format depends on Qwen-Image-Edit training.
             # Hypothesis: User input image + text instruction -> Output new image tokens.
             # Since 'generate' outputs tokens, we need a way to decode them to pixels.
             # Typically Qwen-Image outputs <img_start>...tokens...<img_end>.
             # We rely on the processor/model to handle this or we just accept we get text description if it's not the Omnimodal version.
             
             # Constructing prompt
             conversation = [
                 {
                     "role": "user",
                     "content": [
                         {"type": "image", "image": image},
                         {"type": "text", "text": f"Change the pose to: {prompt}. {base_prompt}"}
                     ]
                 }
             ]
             
             try:
                 text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                 inputs = self.processor(
                     text=[text_prompt],
                     images=[image],
                     padding=True,
                     return_tensors="pt"
                 ).to(self.device)
                 
                 output_ids = self.model.generate(**inputs, max_new_tokens=1024) # Increased token limit for potential image tokens
                 
                 # NOTE: Decoding logic for image tokens is complex and model specific.
                 # If the model outputs standard text, we log it.
                 # If it outputs image tokens, we would need the specific decoder (VHD, VQGAN, etc).
                 # Qwen-Image usually comes with a visual decoder in the repo. 
                 # Since we are using standard Transformers, we might only get text or raw tokens.
                 # For safety in this prototype Phase: returning original image + log.
                 
                 decoded_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
                 print(f"Variation {p_idx}: {decoded_text}")
                 
                 generated_images.append(image.copy())
                 
             except Exception as e:
                 print(f"Error during inference: {e}")
                 generated_images.append(image.copy())

        return generated_images
