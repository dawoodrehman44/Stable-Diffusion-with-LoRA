import torch
import torch.nn as nn
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# --------------------------
# Model Configuration & Setup
# --------------------------

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_chexagent_model():
    device = get_device()

    # Load pre-trained BLIP2 base model
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

    # Replace language model head with identity (remove)
    model.language_model = nn.Identity()

    # Add classification head with 14 classes
    model.classifier = nn.Linear(model.vision_model.config.hidden_size, 14).to(device)

    model = model.to(device)
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

    return model, processor, device

# --------------------------
# Usage Example (commented)
# --------------------------

# if __name__ == "__main__":
#     model, processor, device = build_chexagent_model()
#     print(f"CheXagent model loaded on {device}")
