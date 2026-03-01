import gradio as gr
from modules.mask_generator import MaskGenerator
from modules.expression_editor import ExpressionEditor
from modules.image_cropper import ImageCropper
from modules.utils import resolve_model_path, list_models
import os
import json
import time
import pandas as pd
from PIL import Image

# Initialize Modules
mask_gen = MaskGenerator()
expr_editor = ExpressionEditor()
image_cropper = ImageCropper(mask_gen)

# Settings File Path
SETTINGS_FILE = "config/last_settings.json"

# --- Model Loading Helpers ---
model_map = {}

def get_available_models():
    """
    Scans directories defined in settings.yaml and the default checkpoint path.
    Populates global model_map.
    Returns:
        list of filenames.
    """
    global model_map
    # Correctly navigate configuration structure: models -> expression_editor
    models_config = expr_editor.config.get("models", {})
    settings_config = models_config.get("expression_editor", {})
    
    # 1. Scanned Directories (Primary Source)
    scan_dirs = settings_config.get("model_directories", [])
    if isinstance(scan_dirs, str):
        scan_dirs = [scan_dirs]
        
    print(f"Model directories from config: {scan_dirs}")
    scanned_models = list_models(scan_dirs, extensions=[".safetensors", ".ckpt"])
    
    # 2. Legacy Checkpoint Path (Optional)
    default_ckpt = settings_config.get("checkpoint_path", "")
    model_paths = []
    if default_ckpt and os.path.exists(default_ckpt):
        model_paths.append(default_ckpt)
    
    # Merge and Deduplicate
    all_paths = list(set(scanned_models + model_paths))
    
    # Build Map
    new_map = {}
    for p in all_paths:
        filename = os.path.basename(p)
        new_map[filename] = p
    
    model_map = new_map
    choices = list(model_map.keys())
    choices.sort()
    return choices

def load_selected_model(model_name):
    global model_map
    if not model_name or model_name not in model_map:
        return f"Model not found: {model_name}"
    
    path = model_map[model_name]
    try:
        expr_editor.load_model(checkpoint_path=path)
        return f"Loaded: {model_name}"
    except Exception as e:
        return f"Error loading model: {e}"

# --- Settings Persistence ---
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Cleanup string values to remove extra escapes/literal newlines
            cleaned_data = {}
            for k, v in data.items():
                if isinstance(v, str):
                    # Fix literal "\n" strings that should be real newlines
                    v = v.replace("\\n", "\n")
                    # Fix literal "\r"
                    v = v.replace("\\r", "\r")
                cleaned_data[k] = v
            return cleaned_data
            
        except Exception as e:
            print(f"Error loading settings: {e}")
    return {}

def save_settings(settings):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving settings: {e}")

# Initial Load
saved = load_settings()
DEFAULT_SAM = saved.get("sam_prompt", "face")
DEFAULT_ADJ = saved.get("mask_adjustment", 0)
DEFAULT_SAM_CONF = saved.get("sam_conf", 0.25)
DEFAULT_BASE_POS = saved.get("base_positive_prompt", "high quality, detailed")
DEFAULT_BASE_NEG = saved.get("base_negative_prompt", "lowres, worst quality")
DEFAULT_PREFIX = saved.get("file_prefix", "variation_")
DEFAULT_STRENGTH = saved.get("strength", 0.55)
DEFAULT_GUIDANCE = saved.get("guidance", 7.5)
DEFAULT_STEPS = saved.get("steps", 20)
DEFAULT_CROP_ENABLED = saved.get("crop_enabled", False)
DEFAULT_CROP_PROMPT = saved.get("crop_prompt", "1girl")
DEFAULT_CROP_ADJ = saved.get("crop_adj", 0)
DEFAULT_CROP_CONF = saved.get("crop_conf", 0.25)

# Batch Table Defaults
# Expression Prompt, Filename Suffix
DEFAULT_BATCH_DATA = saved.get("batch_data", [
    ["smile", "smile"],
    ["angry", "angry"],
    ["sad", "sad"],
    ["surprised", "surprised"]
])

# --- Processing Logic ---

# Global Mask Cache
current_mask = None
current_input_image = None
current_mask_image_display = None

def run_mask_generation(img, sam_prompt, adj, conf):
    global current_mask, current_input_image, current_mask_image_display
    if img is None:
        return None
    
    print(f"Generating mask for '{sam_prompt}' with adj {adj}, conf {conf}...")
    mask = mask_gen.generate_mask(img, prompt_text=sam_prompt, dilation_factor=adj, conf=conf)
    
    current_input_image = img
    current_mask = mask # PIL Image (L mode)
    current_mask_image_display = mask
    
    return mask

def run_batch_generation(
    input_img, 
    model_name,
    base_pos, 
    base_neg, 
    batch_df, 
    prefix, 
    strength, 
    guidance, 
    steps,
    enable_crop,
    crop_prompt,
    crop_adj,
    crop_conf,
    sam_prompt,
    mask_adj,
    sam_conf
):
    global current_mask, model_map
    
    if input_img is None:
        return [], "No input image."
        
    # Ensure Model is Loaded
    if not model_name:
        return [], "Error: No model selected."
        
    if model_name not in model_map:
        return [], f"Error: Model '{model_name}' not found in map. Please refresh."
        
    target_path = model_map[model_name]
    
    # Check if we need to load/switch model
    # Accessing internal state of expr_editor (a bit dirty but effective given current structure)
    try:
        # If pipeline not loaded OR current loaded path (if exists) different
        if expr_editor.pipeline is None or \
           not hasattr(expr_editor, 'current_loaded_path') or \
           expr_editor.current_loaded_path != target_path:
               
            print(f"Loading/Switching model to: {model_name}")
            expr_editor.load_model(target_path)
    except Exception as e:
        return [], f"Error loading model: {e}"

    if current_mask is None:
        # Generate default mask if not yet generated manually
        print("Mask not found, generating mask based on UI settings...")
        current_mask = run_mask_generation(input_img, sam_prompt, mask_adj, sam_conf)
    
    # Check output directory
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    # Iterate DataFrame
    if isinstance(batch_df, pd.DataFrame):
        rows = batch_df.values.tolist()
    else:
        rows = batch_df # list of lists
        
    for i, row in enumerate(rows):
        expr_prompt = row[0]
        suffix = row[1]
        
        if not expr_prompt:
            continue
            
        full_pos_prompt = f"{expr_prompt}, {base_pos}" if base_pos else expr_prompt
        
        print(f"Batch {i+1}: {expr_prompt} (File: {suffix})")
        
        guide_size = expr_editor.config.get("face_detailer", {}).get("guide_size", 1024)
        
        try:
            # We already ensured model is loaded via explicit call above.
            # edit_expression (modified) will just check self.pipeline is not None.
            res_img = expr_editor.edit_expression(
                image=input_img,
                mask=current_mask,
                prompt=full_pos_prompt,
                negative_prompt=base_neg,
                strength=strength,
                guidance_scale=guidance,
                num_inference_steps=steps,
                guide_size=guide_size
            )
        except Exception as e:
            print(f"Error processing batch item {i}: {e}")
            continue
        
        # Save
        save_suffix = "".join([c for c in suffix if c.isalnum() or c in ('-', '_')])
        if not save_suffix:
            save_suffix = f"batch_{i}"
            
        filename = f"{prefix}{save_suffix}.png"
        save_path = os.path.join(output_dir, filename)
        
        if os.path.exists(save_path):
            timestamp = int(time.time())
            filename = f"{prefix}{save_suffix}_{timestamp}.png"
            save_path = os.path.join(output_dir, filename)
            
        if enable_crop:
            res_img = image_cropper.crop_image_by_prompt(res_img, crop_prompt, crop_adj, crop_conf)
            
        res_img.save(save_path)
        results.append((res_img, filename))
        
    # Save Settings
    current_settings = {
        "sam_prompt": sam_prompt, 
        "mask_adjustment": mask_adj,
        "sam_conf": sam_conf,
        "base_positive_prompt": base_pos,
        "base_negative_prompt": base_neg,
        "file_prefix": prefix,
        "strength": strength,
        "guidance": guidance,
        "steps": steps,
        "crop_enabled": enable_crop,
        "crop_prompt": crop_prompt,
        "crop_adj": crop_adj,
        "crop_conf": crop_conf,
        "batch_data": rows
    }
    save_settings(current_settings)
        
    return results, f"Generated {len(results)} variations in {output_dir}"


# --- UI Construction ---

# Prepare Model Choices
avail_model_names = get_available_models()
default_model_name = avail_model_names[0] if avail_model_names else None

with gr.Blocks(title="Character Variation Generator v2", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Character Variation Generator")
    
    with gr.Row():
        # LEFT COLUMN: Input & Mask
        with gr.Column(scale=1):
            input_image_box = gr.Image(label="Input Image", type="pil", height=400)
            
            with gr.Row():
                sam_prompt_box = gr.Textbox(label="Segmentation Prompt", value=DEFAULT_SAM, scale=2)
                mask_adj_slider = gr.Slider(minimum=-20, maximum=20, value=DEFAULT_ADJ, step=1, label="Mask Adj", scale=2)
                sam_conf_slider = gr.Slider(minimum=0.01, maximum=1.0, value=DEFAULT_SAM_CONF, step=0.01, label="SAM Conf", scale=2)
                gen_mask_btn = gr.Button("Generate Mask", variant="secondary", scale=1)
            
            mask_display = gr.Image(label="Current Mask", type="pil", interactive=False, height=200)

            gr.Markdown("### Character Cropping Option")
            with gr.Row():
                enable_crop_chk = gr.Checkbox(label="Enable Character Crop After Gen", value=DEFAULT_CROP_ENABLED)
                crop_prompt_box = gr.Textbox(label="Crop SAM Prompt", value=DEFAULT_CROP_PROMPT)
            with gr.Row():
                crop_adj_slider = gr.Slider(minimum=-20, maximum=20, value=DEFAULT_CROP_ADJ, step=1, label="Crop Mask Adj", scale=2)
                crop_conf_slider = gr.Slider(minimum=0.01, maximum=1.0, value=DEFAULT_CROP_CONF, step=0.01, label="Crop SAM Conf", scale=2)
            preview_crop_btn = gr.Button("Preview Crop Range", variant="secondary")
            crop_display = gr.Image(label="Crop Preview", type="pil", interactive=False, height=200)

        # MIDDLE COLUMN: Settings & Batch
        with gr.Column(scale=1):
            # Model Selection
            gr.Markdown("### Model Selection")
            model_dropdown = gr.Dropdown(
                choices=avail_model_names,
                value=default_model_name,
                label="SDXL Checkpoint",
                interactive=True
            )
            load_status = gr.Markdown(f"Current: {default_model_name if default_model_name else 'None'}")
            
            # Base Prompts
            gr.Markdown("### Prompts")
            base_pos_box = gr.TextArea(label="Base Positive Prompt", value=DEFAULT_BASE_POS, lines=3)
            base_neg_box = gr.TextArea(label="Base Negative Prompt", value=DEFAULT_BASE_NEG, lines=3)
            
            # Batch Table
            gr.Markdown("### Batch Expressions")
            batch_table = gr.Dataframe(
                headers=["Expression Prompt", "Filename Suffix"],
                datatype=["str", "str"],
                value=DEFAULT_BATCH_DATA,
                label="Variations List",
                interactive=True,
                row_count=(4, "dynamic"),
                col_count=(2, "fixed")
            )
            
            # File Settings
            prefix_box = gr.Textbox(label="File Prefix", value=DEFAULT_PREFIX)
            
            with gr.Accordion("Advanced Parameters", open=False):
                strength_slider = gr.Slider(minimum=0.0, maximum=1.0, value=DEFAULT_STRENGTH, label="Denoising Strength")
                guidance_slider = gr.Slider(minimum=1.0, maximum=20.0, value=DEFAULT_GUIDANCE, label="Guidance Scale")
                steps_slider = gr.Slider(minimum=1, maximum=100, value=DEFAULT_STEPS, step=1, label="Steps")

            generate_btn = gr.Button("Generate Batch", variant="primary", size="lg")

        # RIGHT COLUMN: Results
        with gr.Column(scale=1):
            gr.Markdown("### Results")
            status_text = gr.Markdown("Ready")
            gallery = gr.Gallery(label="Generated Variations", columns=2, height=600)

    # Events
    
    # Model Change
    model_dropdown.change(
        fn=load_selected_model,
        inputs=[model_dropdown],
        outputs=[load_status]
    )
    
    # Mask Generation
    gen_mask_btn.click(
        fn=run_mask_generation,
        inputs=[input_image_box, sam_prompt_box, mask_adj_slider, sam_conf_slider],
        outputs=[mask_display]
    )
    
    # Batch Generation
    generate_btn.click(
        fn=run_batch_generation,
        inputs=[
            input_image_box,
            model_dropdown,
            base_pos_box,
            base_neg_box,
            batch_table,
            prefix_box,
            strength_slider,
            guidance_slider,
            steps_slider,
            enable_crop_chk,
            crop_prompt_box,
            crop_adj_slider,
            crop_conf_slider,
            sam_prompt_box,
            mask_adj_slider,
            sam_conf_slider
        ],
        outputs=[gallery, status_text]
    )

    # Preview Crop
    preview_crop_btn.click(
        fn=image_cropper.preview_crop_mask,
        inputs=[input_image_box, crop_prompt_box, crop_adj_slider, crop_conf_slider],
        outputs=[crop_display]
    )

if __name__ == "__main__":
    demo.launch()
