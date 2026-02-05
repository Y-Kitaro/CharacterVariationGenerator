import gradio as gr
from modules.pose_generator import PoseGenerator
from modules.upscaler import Upscaler
from modules.mask_generator import MaskGenerator
from modules.expression_editor import ExpressionEditor
import os
from PIL import Image

# Initialize Modules
pose_gen = PoseGenerator()
upscaler = Upscaler()
mask_gen = MaskGenerator()
expr_editor = ExpressionEditor()

def generate_poses(image, base_prompt, pose_prompts_str):
    if image is None:
        return []
    
    pose_prompts = [p.strip() for p in pose_prompts_str.split('\n') if p.strip()]
    if not pose_prompts:
        return []
        
    results = pose_gen.generate_variations(image, base_prompt, pose_prompts)
    return results

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

def detect_face_mask(image):
    if image is None:
        return None
    
    mask = mask_gen.detect_face(image)
    return mask

def generate_expressions(image, mask, base_prompt, expression_prompts_str):
    if image is None or mask is None:
        return []
        
    expr_prompts = [p.strip() for p in expression_prompts_str.split('\n') if p.strip()]
    if not expr_prompts:
        return []
        
    results = []
    for prompt in expr_prompts:
        res = expr_editor.edit_expression(image, mask, prompt, base_prompt)
        results.append((res, prompt)) # Gradio Gallery accepts (image, label) tuples
        
    return results

# Gradio Interface
with gr.Blocks(title="Character Variation Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Character Variation Generator")
    
    with gr.Tabs():
        # --- Step 1: Pose Generation ---
        with gr.Tab("Step 1: Pose Generation"):
            with gr.Row():
                with gr.Column():
                    ref_image = gr.Image(label="Reference Image", type="pil", height=400)
                    base_prompt_input = gr.Textbox(label="Base Prompt (Character Features)", placeholder="e.g. anime girl, blue hair, white dress")
                    pose_prompts_input = gr.Textbox(label="Pose Prompts (One per line)", placeholder="standing\nsitting\nrunning", lines=5)
                    generate_pose_btn = gr.Button("Generate Poses", variant="primary")
                
                with gr.Column():
                    pose_gallery = gr.Gallery(label="Generated Poses", show_label=True, columns=2, height=600, allow_preview=True)
            
            generate_pose_btn.click(
                fn=generate_poses,
                inputs=[ref_image, base_prompt_input, pose_prompts_input],
                outputs=[pose_gallery]
            )

        # --- Step 2: Upscaling ---
        with gr.Tab("Step 2: Upscaling"):
            gr.Markdown("Select an image generated in Step 1 or upload a new one to upscale.")
            with gr.Row():
                with gr.Column():
                    # We can allow user to drag from Step 1 gallery manually or just upload
                    input_image_upscale = gr.Image(label="Input Image", type="pil", height=400)
                    scale_slider = gr.Slider(minimum=2, maximum=4, value=2, step=1, label="Upscale Factor (Generic Resize if not x4)")
                    upscale_btn = gr.Button("Upscale", variant="primary")
                
                with gr.Column():
                    upscaled_image_output = gr.Image(label="Upscaled Image", type="pil", interactive=False)
            
            # Helper to transfer selected image from Step 1 gallery to Step 2 input
            def on_select_pose(evt: gr.SelectData, gallery_data):
                # Retrieve the selected image from gallery data
                # Gallery data is a list of tuples (image_path, label) or just image paths depending on version
                # But Gradio 4 select_data gives index and value.
                # However, getting the PIL image directly from a gallery selection is tricky without re-downloading or state.
                # Simplest is to ask user to download/upload or use a specialized component.
                # For now, user has to download and upload or drag/drop.
                pass 

            upscale_btn.click(
                fn=upscale_image,
                inputs=[input_image_upscale, scale_slider],
                outputs=[upscaled_image_output]
            )

        # --- Step 3: Expression Variation ---
        with gr.Tab("Step 3: Expression Variation"):
            gr.Markdown("Use the upscaled image to generate expression variations.")
            with gr.Row():
                with gr.Column():
                    input_image_expr = gr.Image(label="Input Image (Upscaled)", type="pil", height=400)
                    
                    with gr.Accordion("Mask Settings", open=True):
                        detect_mask_btn = gr.Button("Auto Detect Face Mask (SAM3)")
                        mask_image_output = gr.Image(label="Detected Mask", type="pil", interactive=True) # User can edit mask
                    
                    expr_base_prompt = gr.Textbox(label="Base Prompt", placeholder="same as step 1")
                    expression_list = gr.Textbox(label="Expressions (One per line)", placeholder="smile\nangry\nsurprised\ncrying", lines=5)
                    generate_expr_btn = gr.Button("Generate Expressions", variant="primary")
                
                with gr.Column():
                    expression_gallery = gr.Gallery(label="Expression Variations", columns=3, height=600)

            # Auto-fill base prompt from Step 1 if possible (using global or state?)
            # For simplicity, separate inputs.
            
            detect_mask_btn.click(
                fn=detect_face_mask,
                inputs=[input_image_expr],
                outputs=[mask_image_output]
            )
            
            generate_expr_btn.click(
                fn=generate_expressions,
                inputs=[input_image_expr, mask_image_output, expr_base_prompt, expression_list],
                outputs=[expression_gallery]
            )

if __name__ == "__main__":
    demo.launch()
