import os
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
import torch

# --- Age group categorization ---
def categorize_age(age):
    if age <= 30:
        return 'AGE_GROUP_AGE_0_30'
    elif age <= 50:
        return 'AGE_GROUP_AGE_31_50'
    elif age <= 70:
        return 'AGE_GROUP_AGE_51_70'
    else:
        return 'AGE_GROUP_AGE_71_plus'

# --- Load and preprocess dataset CSVs ---
def load_and_preprocess(train_csv_path, valid_csv_path, dataset_name="chexpert"):
    train_df = pd.read_csv(train_csv_path)
    valid_df = pd.read_csv(valid_csv_path)

    if 'Age' in train_df.columns:
        for df in [train_df, valid_df]:
            df['AGE_GROUP'] = df['Age'].apply(categorize_age)
        train_df = pd.get_dummies(train_df, columns=['AGE_GROUP'])
        valid_df = pd.get_dummies(valid_df, columns=['AGE_GROUP'])

        age_group_cols = ['AGE_GROUP_AGE_0_30', 'AGE_GROUP_AGE_31_50', 'AGE_GROUP_AGE_51_70', 'AGE_GROUP_AGE_71_plus']
        for col in age_group_cols:
            for df in [train_df, valid_df]:
                if col not in df.columns:
                    df[col] = 0

    return train_df, valid_df

# --- Image transforms ---
def get_transforms():
    return transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# --- Dataset class (generalized) ---
class XrayDataset(Dataset):
    def __init__(self, dataframe, transform=None, image_root=None, dataset="chexpert"):
        self.dataframe = dataframe
        self.transform = transform
        self.image_root = image_root
        self.dataset = dataset.lower()

        self.label_cols = {
            "chexpert": ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
                         'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                         'Pleural Effusion', 'Fracture', 'Support Devices'],
            "mimic": ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                      'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                      'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'],
            "cxr14": ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
                      'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
                      'Fibrosis', 'Pleural Thickening'],
            "synthetic": ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
                         'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                         'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

        }[self.dataset]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]

        if self.dataset == "chexpert":
            img_path = item['Path'].replace("", "")
        elif self.dataset == "mimic":
            img_path = item['Path'] 
        elif self.dataset == "cxr14":
            img_path = item['Image Index'],
        elif self.dataset == "synthetic":
            img_path = item['Image Index']
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        img_path = os.path.join(self.image_root, img_path)

        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, IOError, UnidentifiedImageError):
            return None

        if self.transform:
            image = self.transform(image)

        label = item[self.label_cols].values.astype(np.float32)
        return image, label

# --- Safe collate ---
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0)
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels
