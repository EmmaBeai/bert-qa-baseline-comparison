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


def process_answers(raw_answers, position_map):
    cleaned_answers = []
    for ans in raw_answers:
        text, _ = clean_text_with_mapping(ans["text"])
        start = map_position(ans["answer_start"], position_map) if position_map else None
        cleaned_answers.append({
            "text": text,
            "answer_start": start
        })
    return cleaned_answers


# Process SQuAD 2.0 data
samples = []
input_path = Path("data/raw/train-v2.0.json")

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for article in data["data"]:
    for para in article["paragraphs"]:
        context, position_map = clean_text_with_mapping(para["context"])
        for qa in para["qas"]:
            question, _ = clean_text_with_mapping(qa["question"])
            is_impossible = qa.get("is_impossible", False)

            if is_impossible:
                answers = []
                primary_answer = {"text": "", "answer_start": -1}
            else:
                answers = process_answers(qa.get("answers", []), position_map)
                primary_answer = answers[0] if answers else {"text": "", "answer_start": -1}

            # Plausible answers exist for impossible questions (train split)
            plausible_answers = process_answers(qa.get("plausible_answers", []), position_map)

            samples.append({
                "id": qa["id"],
                "question": question,
                "context": context,
                "answer": primary_answer["text"],
                "answer_start": primary_answer["answer_start"] if primary_answer["answer_start"] is not None else -1,
                "is_impossible": is_impossible,
                "answers": answers,
                "plausible_answers": plausible_answers
            })

# Save processed data
output_path = Path("data/processed/clean_squad2_train.json")
output_dir = Path(output_path).parent
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)


