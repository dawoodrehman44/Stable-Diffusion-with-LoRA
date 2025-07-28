# main_train_diffusion.py

import torch
import pandas as pd
from torch.utils.data import DataLoader
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from peft import get_peft_model, LoraConfig
from transformers import CLIPTokenizer
from dataset import CaptionDataset  # Custom dataset handling image-text pairs

# --------------------------
# Hyperparameters and Config
# --------------------------

DATA_CSV = "data/train_captions.csv"
IMAGE_ROOT = "data/images"
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
EPOCHS = 200
IMAGE_SIZE = 512
TEXT_EMBEDDING_LENGTH = 77

# --------------------------
# Training Function
# --------------------------

def train():
    accelerator = Accelerator()
    device = accelerator.device

    # Load data
    df = pd.read_csv(DATA_CSV)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    dataset = CaptionDataset(df, IMAGE_ROOT, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load base diffusion model
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)
    pipe.to(device)
    unet = pipe.unet

    # Apply LoRA adaptation
    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["to_q", "to_k", "to_v"],
        lora_dropout=0.1,
        bias="none"
    )
    unet = get_peft_model(unet, lora_config)

    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Prepare for training
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    # Training loop
    for epoch in range(EPOCHS):
        unet.train()
        total_loss = 0
        for step, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            # Forward + loss computation (assuming UNet and caption-to-image setup)
            outputs = unet(pixel_values, encoder_hidden_states=input_ids)
            loss = outputs.loss

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        accelerator.print(f"Epoch [{epoch + 1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

        # Save checkpoint
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(unet)
        torch.save(unwrapped_model.state_dict(), f"checkpoints/diffusion_lora_epoch_{epoch + 1}.pt")

if __name__ == "__main__":
    train()
