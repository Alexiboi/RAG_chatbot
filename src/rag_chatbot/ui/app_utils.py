def load_params_from_txt(file_path: str) -> str:
    """Load text content from a .txt file and split on '='."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print("No content file found at: ", file_path)
        return {}

    lines = content.splitlines()
    params = {}
    for line in lines:
        if '=' in line:
            key, value = line.split('=', 1)
            params[key.strip()] = value.strip()
    
    return params
