import json
from pathlib import Path

def clean_text_with_mapping(text):
    """
    Clean text by normalizing whitespace and return both cleaned text
    and a mapping from original positions to cleaned positions.
    
    Note: We do NOT strip leading/trailing whitespace to preserve absolute indices.
    
    Returns:
        cleaned_text: str - the cleaned text
        position_map: dict - mapping from original index to cleaned index
                          (None if original position was removed)
    """
    cleaned = []
    position_map = {}
    
    i = 0  # index in original text
    j = 0  # index in cleaned text
    in_whitespace = False
    
    # Map the text to the cleaned text
    while i < len(text):
        if text[i].isspace():
            if not in_whitespace:
                # Map the first whitespace character to a single space
                cleaned.append(' ')
                position_map[i] = j
                j += 1
                in_whitespace = True
            else:
                # Remove subsequent whitespace characters
                position_map[i] = None
            i += 1
        else:
            # keep non-whitespace characters
            cleaned.append(text[i])
            position_map[i] = j
            j += 1
            in_whitespace = False
            i += 1
    
    return ''.join(cleaned), position_map

def map_position(original_pos, position_map):
    """
    Map an original position to cleaned position.
    If the position is removed, find the next valid position.
    """
    mapped = position_map.get(original_pos)
    if mapped is not None:
        return mapped
    
    # If the position is removed, find the next valid position
    for offset in range(1, min(10, len(position_map) - original_pos)):
        check_pos = original_pos + offset
        if position_map.get(check_pos) is not None:
            return position_map[check_pos]
    
    return None

# Process SQuAD data
samples = []
input_path = "raw/dev-v1.1.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for article in data["data"]:
    for para in article["paragraphs"]:
        context, position_map = clean_text_with_mapping(para["context"])
        for qa in para["qas"]:
            q, _ = clean_text_with_mapping(qa["question"])
            a = qa["answers"][0]
            ans, _ = clean_text_with_mapping(a["text"])
            original_start = a["answer_start"]
            start = map_position(original_start, position_map)
            samples.append({
                "id": qa["id"],
                "question": q,
                "context": context,
                "answer": ans,
                "answer_start": start
            })

# Save processed data
output_path = "data/processed/clean_squad_dev.json"
output_dir = Path(output_path).parent
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

