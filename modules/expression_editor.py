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
        
        self.checkpoint_path = None
        self.device = self.config["device"]
        self.pipeline = None
        self.default_denoising = self.config["processing"].get("default_denoising_strength", 0.55)

    def load_model(self, checkpoint_path=None):
        target_path = checkpoint_path if checkpoint_path else self.checkpoint_path
        
        # Check if we are already loaded with the correct model
        if hasattr(self, 'current_loaded_path') and self.current_loaded_path == target_path and self.pipeline is not None:
            return

        # If different model loaded, unload first
        if self.pipeline is not None:
             self.unload_model()

        if not target_path or not os.path.exists(target_path):
             print(f"Warning: SDXL Checkpoint not found at {target_path}.")
             return
        else:
            print(f"Loading SDXL Inpainting model from {target_path}...")
            self.pipeline = StableDiffusionXLInpaintPipeline.from_single_file(
                target_path,
                torch_dtype=torch.float16,
                use_safetensors=True
            ).to(self.device)
        
        self.current_loaded_path = target_path
        
        # Initialize Compel for Long Prompt weighting
        try:
            from compel import Compel, ReturnedEmbeddingsType
            print("Initializing Compel for SDXL Long Prompts...")
            self.compel = Compel(
                tokenizer=[self.pipeline.tokenizer, self.pipeline.tokenizer_2],
                text_encoder=[self.pipeline.text_encoder, self.pipeline.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                truncate_long_prompts=False
            )
        except ImportError:
            print("Warning: Compel library not found. Long prompts/weighting might not work as expected.")
            self.compel = None
            
        print(f"ExpressionEditor model loaded: {target_path}")

    def unload_model(self):
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            torch.cuda.empty_cache()

    def get_pipeline_embeds(self, prompt, negative_prompt, device):
        """
        Helper to get long prompt embeds for SDXL (WebUI style chunking).
        """
        import math
        
        tokenizers = [self.pipeline.tokenizer, self.pipeline.tokenizer_2]
        text_encoders = [self.pipeline.text_encoder, self.pipeline.text_encoder_2]
        
        # 1. Tokenize (raw, no special tokens) with explicit truncation=False
        # Use simple tokenizer call to ensure we get full list of IDs
        p_ids = [t(prompt, add_special_tokens=False, truncation=False)["input_ids"] for t in tokenizers]
        n_ids = [t(negative_prompt, add_special_tokens=False, truncation=False)["input_ids"] for t in tokenizers]
        
        # 2. Determine max chunks (75 tokens per chunk allows +2 for BOS/EOS)
        max_len = 0
        for ids in p_ids + n_ids:
            max_len = max(max_len, len(ids))
        
        total_chunks = math.ceil(max_len / 75) if max_len > 0 else 1
        
        prompt_embeds_parts = []
        neg_embeds_parts = []
        pooled_prompt_embeds = None
        neg_pooled_prompt_embeds = None
        
        for k, (tokenizer, text_encoder) in enumerate(zip(tokenizers, text_encoders)):
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            eos_id = tokenizer.eos_token_id
            bos_id = tokenizer.bos_token_id
            
            def build_chunks(tokens):
                chunks = []
                for i in range(total_chunks):
                    start = i * 75
                    end = start + 75
                    chunk_tokens = tokens[start:end]
                    
                    # Add specialized tokens: [BOS] + tokens + [EOS]
                    built = [bos_id] + chunk_tokens + [eos_id]
                    
                    # Pad to 77
                    if len(built) < 77:
                        built += [pad_id] * (77 - len(built))
                    
                    # Truncate if strictly > 77 (safety)
                    built = built[:77]
                    
                    chunks.append(torch.tensor(built, dtype=torch.long, device=device))
                return torch.stack(chunks)
            
            batch_p = build_chunks(p_ids[k])
            batch_n = build_chunks(n_ids[k])
            
            with torch.no_grad():
                out_p = text_encoder(batch_p, output_hidden_states=True)
                out_n = text_encoder(batch_n, output_hidden_states=True)
                
                # Hidden states (penultimate)
                # SDXL generally uses hidden_states[-2]
                hidden_p = out_p.hidden_states[-2]
                hidden_n = out_n.hidden_states[-2]
                
                # Flatten chunks into sequence: (chunks, 77, dim) -> (1, chunks*77, dim)
                prompt_embeds_parts.append(hidden_p.view(1, -1, hidden_p.shape[-1]))
                neg_embeds_parts.append(hidden_n.view(1, -1, hidden_n.shape[-1]))
                
                # Pooled (only from Encoder 2)
                # Use pooled embedding from the FIRST chunk
                # CLIPTextModelWithProjection has text_embeds
                if k == 1 and pooled_prompt_embeds is None:
                    pooled_prompt_embeds = out_p.text_embeds[0].unsqueeze(0)
                    neg_pooled_prompt_embeds = out_n.text_embeds[0].unsqueeze(0)

        # Concat embeddings from both encoders along feature dim
        prompt_embeds = torch.cat(prompt_embeds_parts, dim=-1)
        neg_embeds = torch.cat(neg_embeds_parts, dim=-1)
        
        return prompt_embeds, neg_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds

    def edit_expression(self, image: Image.Image, mask: Image.Image, prompt: str, negative_prompt: str = "", strength: float = 0.55, guidance_scale: float = 7.5, num_inference_steps: int = 20, guide_size: int = 1024, feather: int = 5) -> Image.Image:
        from PIL import ImageFilter
        import cv2
        import numpy as np

        if self.pipeline is None:
             # Try default load if not loaded, but warn
             print("Pipeline not loaded in edit_expression, attempting default load...")
             self.load_model()
             if self.pipeline is None:
                 raise RuntimeError("Model failed to load.")
        
        print(f"Editing expression with prompt: {prompt[:50]}..., strength: {strength}")
        
        # Encode Long Prompts (Custom SDXL Logic)
        (
             prompt_embeds,
             neg_embeds,
             pooled_embeds,
             neg_pooled_embeds
        ) = self.get_pipeline_embeds(prompt, negative_prompt, self.device)
        
        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Ensure mask is L mode and same size
        if mask.mode != "L":
            mask = mask.convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)
            
        final_image = image.copy()
        original_width, original_height = image.size
        
        # Find Connected Components in the mask (to handle multiple faces or isolated regions)
        mask_np = np.array(mask)
        # Binarize just in case
        _, thresh = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
        
        # Dilation to merge fragmented components (e.g. eyes, mouth detected separately)
        # Use large kernel to bridge gaps within a face
        kernel = np.ones((20, 20), np.uint8)
        dilated_mask = cv2.dilate(thresh, kernel, iterations=3)
        
        # Find components on DILATED mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated_mask, connectivity=8)
        
        print(f"Mask analysis: Found {num_labels - 1} face regions.")
        
        generator = torch.Generator(device=self.device).manual_seed(42)

        for i in range(1, num_labels):
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            
            # Margin strategy from sam_test.py
            margin = 32
            x_min = max(0, x - margin)
            y_min = max(0, y - margin)
            x_max = min(original_width, x + w + margin)
            y_max = min(original_height, y + h + margin)
            
            bbox = (x_min, y_min, x_max, y_max)
            print(f"Processing Face Region {i}: BBox {bbox}")
            
            # Crop Original Image
            cropped_img = image.crop(bbox)
            
            # Create mask for this specific component
            cropped_mask = mask.crop(bbox)
            
            target_size = (guide_size, guide_size)
            img_resized = cropped_img.resize(target_size, Image.LANCZOS)
            mask_resized = cropped_mask.resize(target_size, Image.LANCZOS)

            # Inpaint
            output_resized = self.pipeline(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=neg_embeds,
                pooled_prompt_embeds=pooled_embeds,
                negative_pooled_prompt_embeds=neg_pooled_embeds,
                image=img_resized,
                mask_image=mask_resized,
                strength=strength, 
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                width=guide_size,
                height=guide_size,
                generator=generator
            ).images[0]
            
            # Resize back
            output_crop = output_resized.resize(cropped_img.size, Image.LANCZOS)
            
            # Feathering composite
            # Create a soft mask from the cropped mask
            # We used binary mask for crop, but let's re-crop the original mask for feathering
            
            # Apply feather to the mask used for compositing
            # We can use the resized mask or the original cropped mask
            # Better to use the original resolution mask for final composite
            mask_for_paste = mask.crop(bbox)
            if feather > 0:
                 mask_for_paste = mask_for_paste.filter(ImageFilter.GaussianBlur(feather))
            
            final_image.paste(output_crop, (x_min, y_min), mask=mask_for_paste)
            
        print("Expression editing complete.")
        return final_image
