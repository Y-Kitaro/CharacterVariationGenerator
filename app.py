import gradio as gr
from modules.mask_generator import MaskGenerator
from modules.expression_editor import ExpressionEditor
from modules.utils import resolve_model_path, list_models
import os
import json
import time
from PIL import Image

# Initialize Modules
mask_gen = MaskGenerator()
expr_editor = ExpressionEditor()

# Settings File Path
SETTINGS_FILE = "config/last_settings.json"

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading settings: {e}")
    return {}

def save_settings(settings):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=4)
    except Exception as e:
        print(f"Error saving settings: {e}")

# Load initial settings to populate defaults
saved = load_settings()
DEFAULT_SAM_PROMPT = saved.get("sam_prompt", "face")
DEFAULT_PROMPT = saved.get("prompt", "smile, high quality")
DEFAULT_NEG_PROMPT = saved.get("negative_prompt", "lowres, bad anatomy, bad hands, cropped, worst quality")
DEFAULT_STRENGTH = saved.get("strength", 0.50)
DEFAULT_GUIDANCE = saved.get("guidance", 5.0)
DEFAULT_STEPS = saved.get("steps", 20)
DEFAULT_MASK_ADJ = saved.get("mask_adjustment", 0)

# Get guide_size from config/settings.yaml (via module config)
# We access the loaded config from one of the modules
DEFAULT_GUIDE_SIZE = expr_editor.config.get("face_detailer", {}).get("guide_size", 1024)

def run_process(input_image, sam_prompt, inpaint_prompt, negative_prompt, strength, guidance, steps, mask_adj):
    if input_image is None:
        return None, None
    
    # Save current settings
    current_settings = {
        "sam_prompt": sam_prompt,
        "prompt": inpaint_prompt,
        "negative_prompt": negative_prompt,
        "strength": strength,
        "guidance": guidance,
        "steps": steps,
        "mask_adjustment": mask_adj
    }
    save_settings(current_settings)
    
    print(f"Generating mask for '{sam_prompt}'...")
    # Generate Mask
    mask = mask_gen.generate_mask(input_image, prompt_text=sam_prompt, dilation_factor=mask_adj)
    
    print("Editing expression...")
    # Edit Expression
    # guide_size is read from settings.yaml (loaded in expr_editor.config)
    # We can re-read it to ensure updates in yaml are caught if we want, but usually config is static per run.
    guide_size = expr_editor.config.get("face_detailer", {}).get("guide_size", 1024)
    
    result_image = expr_editor.edit_expression(
        image=input_image,
        mask=mask,
        prompt=inpaint_prompt,
        negative_prompt=negative_prompt,
        strength=strength,
        guidance_scale=guidance,
        num_inference_steps=steps,
        guide_size=guide_size
    )
    
    return result_image, mask

# Gradio UI
with gr.Blocks(title="Character Variation Generator (SAM3 + SDXL)", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Character Variation Generator")
    gr.Markdown("Based on SAM3 for segmentation and SDXL (Compel) for inpainting.")
    
    with gr.Row():
        # Column 1: Input
        with gr.Column(scale=1):
            input_image_box = gr.Image(label="Input Image", type="pil", height=500)

        # Column 2: Settings
        with gr.Column(scale=1):
            with gr.Accordion("Segmentation Settings", open=True):
                sam_prompt_box = gr.Textbox(label="SAM Prompt", value=DEFAULT_SAM_PROMPT)
                mask_adj_slider = gr.Slider(minimum=-20, maximum=20, value=DEFAULT_MASK_ADJ, step=1, label="Mask Adjustment (Dilate/Erode)")
            
            # Inpainting Settings directly visible
            prompt_box = gr.TextArea(label="Inpaint Prompt", value=DEFAULT_PROMPT, lines=8)
            neg_prompt_box = gr.TextArea(label="Negative Prompt", value=DEFAULT_NEG_PROMPT, lines=5)
            
            with gr.Group():
                strength_slider = gr.Slider(minimum=0.0, maximum=1.0, value=DEFAULT_STRENGTH, step=0.01, label="Denoising Strength")
                guidance_slider = gr.Slider(minimum=1.0, maximum=20.0, value=DEFAULT_GUIDANCE, step=0.5, label="Guidance Scale (CFG)")
                steps_slider = gr.Slider(minimum=1, maximum=100, value=DEFAULT_STEPS, step=1, label="Steps")
            
            run_btn = gr.Button("Generate", variant="primary", size="lg")
            
        # Column 3: Output
        with gr.Column(scale=1):
            output_image_box = gr.Image(label="Result", type="pil", interactive=False, height=500)
            
            with gr.Accordion("Generated Mask", open=False):
                mask_image_box = gr.Image(label="Mask", type="pil", interactive=False)

    run_btn.click(
        fn=run_process,
        inputs=[
            input_image_box,
            sam_prompt_box,
            prompt_box,
            neg_prompt_box,
            strength_slider,
            guidance_slider,
            steps_slider,
            mask_adj_slider
        ],
        outputs=[output_image_box, mask_image_box]
    )

if __name__ == "__main__":
    demo.launch()
