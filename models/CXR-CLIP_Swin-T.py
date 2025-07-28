import torch

# --------------------------
# Model Configuration
# --------------------------

model_config = {
    "load_backbone_weights": "checkpoints/cxrclip_mc/swint_mcc.pt",  # ✅ Absolute path
    "freeze_backbone_weights": False,
    "image_encoder": {
        "name": "swin",
        "swin_type": "swin_tiny",
        "pretrained": True,
        "source": "cxr_clip"
    },
    "classifier": {
        "config": {
            "name": "linear",
            "n_class": 14 # can be changed with experiments
        }
    }
}


# --------------------------
# Model Instantiation
# --------------------------

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model():
    from image_classification import CXRClassification  # Make sure this is in your PYTHONPATH
    device = get_device()
    model = CXRClassification(model_config=model_config, model_type="swin")
    model = model.to(device)
    return model, device

# --------------------------
# Usage Example (commented)
# --------------------------

# if __name__ == "__main__":
#     model, device = build_model()
#     print(f"Model loaded on device: {device}")
