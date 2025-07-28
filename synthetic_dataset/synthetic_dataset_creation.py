import os
import csv
import random
import torch
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model

import warnings
warnings.filterwarnings("ignore")


NUM_IMAGES = # as per your requirements

DISEASES = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]

SEXES = ["M", "F"]
RACES = ["white", "black", "asian"]
AGE_GROUPS = list(range(10, 70))  # realistic ages 10-90
AP_PA = ["AP", "PA"]
FRONTAL_LATERAL = ["Frontal"]  # Only frontal now

OUTPUT_DIR = ""
IMAGES_BASE_DIR = os.path.join(OUTPUT_DIR, "train")
CSV_PATH = os.path.join(OUTPUT_DIR, "generated_dataset.csv")

os.makedirs(IMAGES_BASE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading pipeline...")
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.safety_checker = None
pipe.to(DEVICE)

lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v"],
    lora_dropout=0.1,
    bias="none"
)
pipe.unet = get_peft_model(pipe.unet, lora_config)

checkpoint_path = os.path.join(
    "example_checkpoint_path"
)

print(f"Loading LoRA weights from {checkpoint_path} ...")
pipe.unet.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
pipe.unet.eval()


def generate_prompt(disease, age, sex, race, ap_pa):
    disease_str = disease.lower() if disease != "No Finding" else "no significant findings"
    gender_str = "male" if sex == "M" else "female"
    race_str = race.lower()
    return f"Chest X-ray of a {age}-year-old {race_str} {gender_str} patient showing {disease_str}"


with open(CSV_PATH, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    csv_writer.writerow([
        "Path", "Sex", "Age", "Race", "AP/PA", "Frontal/Lateral", *DISEASES
    ])

    for i in tqdm(range(NUM_IMAGES), desc="Generating images"):
        patient_id = f"patient{random.randint(1, 99999):05d}"
        study_id = f"study{random.randint(1, 20)}"

        disease = random.choice(DISEASES)
        sex = random.choice(SEXES)
        race = random.choice(RACES)
        age = random.choice(AGE_GROUPS)
        ap_pa = random.choice(AP_PA)
        frontal_lateral = "Frontal"  # fixed

        prompt = generate_prompt(disease, age, sex, race, ap_pa)

        image = pipe(prompt=prompt).images[0]

        save_dir = os.path.join(IMAGES_BASE_DIR, patient_id, study_id)
        os.makedirs(save_dir, exist_ok=True)

        filename = "view1_frontal.png"
        img_path = os.path.join(save_dir, filename)
        image.save(img_path)

        rel_path = os.path.relpath(img_path, OUTPUT_DIR)

        label_vector = [1 if d == disease else 0 for d in DISEASES]

        csv_writer.writerow([rel_path, sex, age, race, ap_pa, frontal_lateral, *label_vector])

print(f"Dataset generation complete! Saved at: {OUTPUT_DIR}")
