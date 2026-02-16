import os

def list_models(directories, extensions=[".safetensors", ".ckpt"]):
    """
    Scans a list of directories for files with specific extensions.
    
    Args:
        directories (list): List of directory paths to scan.
        extensions (list): List of file extensions to look for.
        
    Returns:
        list: A list of absolute file paths found.
    """
    model_files = []
    
    if isinstance(directories, str):
        directories = [directories]
        
    for directory in directories:
        # Resolve relative paths relative to CWD if needed, or keep absolute
        # If relative, os.path.abspath might be safer
        if not os.path.isabs(directory):
             directory = os.path.abspath(directory)
             
        if not os.path.exists(directory):
            print(f"Warning: Model directory not found: {directory}")
            continue
            
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    model_files.append(os.path.join(root, file))
                    
    return model_files
