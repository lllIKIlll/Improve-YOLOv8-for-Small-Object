import re
from collections import OrderedDict

ATTENTION_LAYER_INDICES = {5, 8}


def build_index_mapping(attention_indices=None, pretrained_layers=23):
    """
    Build mapping from pretrained (yolov8n) layer indices to custom model layer indices.

    yolov8n has 23 layers (10 backbone + 12 head + 1 Detect, indices 0-22).
    CBAM/CA are inserted at layer 5 and 8 in the backbone, increasing total layers to 25 (indices 0-24).
    """
    if attention_indices is None:
        attention_indices = ATTENTION_LAYER_INDICES

    attention_indices = sorted(attention_indices)
    total_target_layers = pretrained_layers + len(attention_indices)
    mapping = {}
    src_idx = 0

    for target_idx in range(total_target_layers):
        if target_idx in attention_indices:
            continue
        mapping[src_idx] = target_idx
        src_idx += 1

    return mapping


def remap_state_dict(pretrained_sd, index_mapping):
    """Rename keys in pretrained state_dict according to index_mapping."""
    new_sd = OrderedDict()
    pattern = re.compile(r"^(model\.)(\d+)(\..*)")

    for key, value in pretrained_sd.items():
        match = pattern.match(key)
        if not match:
            new_sd[key] = value
            continue

        prefix, layer_idx_str, suffix = match.groups()
        layer_idx = int(layer_idx_str)

        if layer_idx in index_mapping:
            new_key = f"{prefix}{index_mapping[layer_idx]}{suffix}"
            new_sd[new_key] = value

    return new_sd


def transfer_pretrained_weights(custom_model, pretrained_path="yolov8n.pt"):
    from ultralytics import YOLO
    import torch

    pretrained = YOLO(pretrained_path)
    pretrained_sd = pretrained.model.state_dict()
    custom_sd = custom_model.model.state_dict()

    index_mapping = build_index_mapping()
    remapped_sd = remap_state_dict(pretrained_sd, index_mapping)

    loaded = 0
    for key in custom_sd:
        if key in remapped_sd and custom_sd[key].shape == remapped_sd[key].shape:
            custom_sd[key] = remapped_sd[key]
            loaded += 1

    custom_model.model.load_state_dict(custom_sd)
    total = len(custom_sd)
    print(f"Weights transferred successfully: {loaded}/{total} items transferred "
          f"({loaded/total*100:.1f}%)")
    return loaded
