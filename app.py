import gradio as gr
from modules.upscaler import Upscaler
from modules.mask_generator import MaskGenerator
from modules.expression_editor import ExpressionEditor
import os
from PIL import Image
import ast
import time

# Initialize Modules
upscaler = Upscaler()
mask_gen = MaskGenerator()
expr_editor = ExpressionEditor()

def upscale_image(image, scale_factor):
    if image is None:
        return None
    
    # Ensure scale_factor is float
    try:
        scale = float(scale_factor)
    except:
        scale = 2.0
        
    result = upscaler.upscale(image, scale)
    return result

def detect_face_mask(image, points_str, mask_adjustment):
    if image is None:
        return None
    
    points = None
    if points_str:
        try:
            parsed = ast.literal_eval(points_str)
            if isinstance(parsed, list):
                points = parsed
        except:
            print("Failed to parse points string, using default.")
    
    if not points: 
        points = None
        
    mask = mask_gen.generate_mask(image, points=points, dilation_factor=mask_adjustment)
    return mask

def generate_expressions(image, mask, base_prompt, expression_data, strength, guidance, steps, file_prefix, model_path):
    if image is None or mask is None:
        return []

    # Ensure the correct model is loaded
    if model_path:
        expr_editor.load_model(model_path)
        
    # expression_data is a list of lists or pandas df. Gradio passes list of lists by default for 'array' type? 
    # Let's check type or assume list of lists if type='array'
    # Row format: [Name, Prompt]
    
    # Filter empty rows
    valid_rows = []
    if hasattr(expression_data, 'values'): # It's a dataframe
        expression_data = expression_data.values.tolist()
        
    for row in expression_data:
        if row and len(row) >= 2 and (row[0] or row[1]):
            valid_rows.append(row)
            
    if not valid_rows:
        return []
    
    # Ensure output directory exists
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
        
    results = []
    timestamp = int(time.time())
    
    for i, row in enumerate(valid_rows):
        expr_name = str(row[0]).strip()
        expr_prompt = str(row[1]).strip()
        
        # If prompt is empty, use name? Or skip? Let's use name as prompt if prompt empty.
        if not expr_prompt:
            expr_prompt = expr_name
            
        print(f"Generating {expr_name}: {expr_prompt}")
        
        res = expr_editor.edit_expression(
            image, 
            mask, 
            expr_prompt, 
            base_prompt, 
            strength=strength, 
            guidance_scale=guidance, 
            num_inference_steps=steps
        )
        
        # Save image
        # Filename: prefix_timestamp_name.png
        safe_name = "".join(x for x in expr_name if x.isalnum() or x in " -_")
        if not safe_name:
            safe_name = f"expr_{i}"
            
        filename = f"{file_prefix}_{timestamp}_{safe_name}.png"
        file_path = os.path.join(output_dir, filename)
        res.save(file_path)
        
        # Gallery label: Name
        results.append((res, expr_name)) 
        
    return results

# Gradio Interface
with gr.Blocks(title="Character Variation Generator v1.0", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Character Variation Generator v1.0")
    gr.Markdown("Upload an image, optionally upscale it, and then generate expression variations.")
    
    with gr.Tabs():
        # --- Step 1: Image Upload & Upscale ---
        with gr.Tab("Step 1: Image Upload & Upscale"):
            with gr.Row():
                with gr.Column():
                    input_image_upscale = gr.Image(label="Input Image", type="pil", height=400)
                    scale_slider = gr.Slider(minimum=2, maximum=4, value=2, step=1, label="Upscale Factor (Generic Resize if not x4)")
                    upscale_btn = gr.Button("Upscale", variant="primary")
                
                with gr.Column():
                    upscaled_image_output = gr.Image(label="Upscaled / Processed Image", type="pil", interactive=False)
            
            upscale_btn.click(
                fn=upscale_image,
                inputs=[input_image_upscale, scale_slider],
                outputs=[upscaled_image_output]
            )

        # --- Step 2: Expression Variation ---
        with gr.Tab("Step 2: Expression Variation"):
            gr.Markdown("Use the upscaled image from Step 1 (or upload new) to generate expression variations.")
            
            with gr.Row():
                # Column 1: Input & Masking
                with gr.Column(scale=1):
                    # Link Step 1 Output to Step 2 Input? 
                    # For now user can drag and drop or we can add a button to transfer.
                    # Or just use the same image component technically? Separation is cleaner.
                    input_image_expr = gr.Image(label="Input Image - Click to Select Mask Point", type="pil", height=500)
                    
                    gr.Markdown("### Mask Settings")
                    gr.Markdown("Click on the face to set SAM point.")
                    points_state = gr.State([])
                    points_text = gr.Textbox(label="Selected Points (x, y)", interactive=True, placeholder="[]")
                    mask_adj_slider = gr.Slider(minimum=-20, maximum=20, value=0, step=1, label="Mask Adjustment (Dilate/Erode)")
                    
                    with gr.Row():
                        detect_mask_btn = gr.Button("Generate Mask", variant="secondary")
                        clear_points_btn = gr.Button("Clear Points")
                        
                    mask_image_output = gr.Image(label="Detected Mask", type="pil", interactive=True, height=250) 
                
                # Column 2: Settings & Prompts
                with gr.Column(scale=1):
                    gr.Markdown("### Prompts & Settings")
                    expr_base_prompt = gr.Textbox(label="Base Prompt", placeholder="blue hair, white dress, anime style")
                    
                    gr.Markdown("Define Expressions (Name -> Filename, Prompt -> Instruction)")
                    expression_data = gr.Dataframe(
                        headers=["Name", "Prompt"],
                        datatype=["str", "str"],
                        value=[["smile", "smile"], ["angry", "angry face"], ["cry", "crying"]],
                        col_count=(2, "fixed"),
                        type="array",
                        label="Expressions List",
                        interactive=True
                    )
                    
                    with gr.Accordion("Advanced Settings", open=True):
                        # Model Selection
                        model_dirs = expr_editor.config["models"]["expression_editor"].get("model_directories", ["assets/models/sdxl"])
                        # Configure default model path
                        default_model_path = expr_editor.config["models"]["expression_editor"]["checkpoint_path"]
                        
                        from modules.utils import list_models
                        
                        def get_model_list():
                            models = list_models(model_dirs)
                            # Ensure default is in list if exists
                            if default_model_path and os.path.exists(default_model_path) and default_model_path not in models:
                                models.insert(0, default_model_path)
                            return models

                        current_models = get_model_list()
                        
                        with gr.Row():
                            model_selector = gr.Dropdown(
                                label="Expression Model", 
                                choices=current_models, 
                                value=default_model_path if default_model_path in current_models else (current_models[0] if current_models else None),
                                interactive=True,
                                scale=3
                            )
                            refresh_models_btn = gr.Button("🔄", variant="secondary", scale=0, size="sm")

                        def refresh_models_list():
                             new_models = get_model_list()
                             return gr.update(choices=new_models)

                        refresh_models_btn.click(fn=refresh_models_list, outputs=[model_selector])

                        strength_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.55, step=0.01, label="Denoising Strength")
                        guidance_slider = gr.Slider(minimum=1.0, maximum=20.0, value=7.5, step=0.5, label="Guidance Scale")
                        steps_slider = gr.Slider(minimum=10, maximum=100, value=25, step=1, label="Inference Steps")
                        file_prefix_input = gr.Textbox(label="Output Filename Prefix", value="char_var")
                    
                    generate_expr_btn = gr.Button("Generate Expressions", variant="primary", size="lg")

                # Column 3: Output Interaction
                with gr.Column(scale=1):
                    gr.Markdown("### Results")
                    expression_gallery = gr.Gallery(label="Output Expressions", columns=2, height=800)

            # --- Event Handlers for Masking ---
            def on_image_click(evt: gr.SelectData, current_points):
                point = [evt.index[0], evt.index[1]]
                if current_points is None:
                    current_points = []
                current_points.append(point)
                return current_points, str(current_points)
            
            input_image_expr.select(
                fn=on_image_click,
                inputs=[points_state],
                outputs=[points_state, points_text]
            )
            
            def on_clear_points():
                return [], "[]"
            
            clear_points_btn.click(on_clear_points, outputs=[points_state, points_text])

            detect_mask_btn.click(
                fn=detect_face_mask,
                inputs=[input_image_expr, points_text, mask_adj_slider],
                outputs=[mask_image_output]
            )
            
            generate_expr_btn.click(
                fn=generate_expressions,
                inputs=[
                    input_image_expr, 
                    mask_image_output, 
                    expr_base_prompt, 
                    expression_data,
                    strength_slider,
                    guidance_slider,
                    steps_slider,
                    file_prefix_input,
                    model_selector
                ],
                outputs=[expression_gallery]
            )

if __name__ == "__main__":
    demo.launch()
