import json


def load_mappings(path):
    with open(path) as f:
        data = json.load(f)

    if "vocab" not in data or "state" not in data:
        raise ValueError("Mapping file must contain 'vocab' and 'state' keys.")

    return data["vocab"], data["state"]


def invert_mapping(mapping):
    return {v: k for k, v in mapping.items()}


def postprocess(input_path, mapping_path, output_path):
    vocab_map, state_map = load_mappings(mapping_path)
    id_to_token = invert_mapping(vocab_map)
    id_to_state = invert_mapping(state_map)

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line_num, line in enumerate(fin, 1):
            try:
                ex = json.loads(line)
                token_ids = ex["token"]
                state_id = ex["state"]
                tokens = [id_to_token[tok] for tok in token_ids]
                state = id_to_state[state_id]
                json.dump({"token": tokens, "state": state}, fout)
                fout.write("\n")
            except Exception as e:
                raise RuntimeError(f"Error processing line {line_num}: {e}")
