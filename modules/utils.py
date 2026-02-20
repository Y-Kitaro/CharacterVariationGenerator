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
        if not os.path.isabs(directory):
             directory = os.path.abspath(directory)
             
        if not os.path.exists(directory):
            print(f"Warning: Model directory not found: {directory}")
            continue
            
        print(f"Scanning directory: {directory}")
        for root, _, files in os.walk(directory):
            print(f"  Root: {root}, Files: {files}")
            for file in files:
                if any(file.lower().endswith(ext.lower()) for ext in extensions):
                    full_path = os.path.join(root, file)
                    print(f"  Found model: {full_path}")
                    model_files.append(full_path)
                    
    return model_files

def resolve_model_path(path, default_path, label="model"):
    """
    Resolves a model path. If it's relative, it's converted to absolute.
    If it exists, return it. Otherwise returns default_path.
    """
    if not path:
        return default_path
    
    abs_path = os.path.abspath(path) if not os.path.isabs(path) else path
    
    if os.path.exists(abs_path):
        print(f"Using local {label} at: {abs_path}")
        return abs_path
    else:
        print(f"Warning: {label} not found at {path}, falling back to {default_path}")
        return default_path
