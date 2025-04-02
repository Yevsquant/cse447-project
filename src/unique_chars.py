import json

def get_unique_chars(work_dir):
    file_path = "".join([work_dir, "/idx_to_char.json"])
    # Load the idx_to_char dictionary from the JSON file
    with open(file_path, 'r', encoding='utf-8') as f:
        idx_to_char = json.load(f)

    # Convert keys back to integers (JSON saves keys as strings)
    idx_to_char = {int(k): v for k, v in idx_to_char.items()}

    num_classes = len(idx_to_char)
    return idx_to_char, num_classes