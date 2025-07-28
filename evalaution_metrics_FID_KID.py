import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.linalg import sqrtm
from PIL import Image
import torch
from torchvision import transforms, models
from sklearn.metrics.pairwise import polynomial_kernel

class FIDEvaluator:
    def __init__(self, original_root, generated_root, original_csv, generated_csv, device=None):
        self.original_root = original_root
        self.generated_root = generated_root
        self.original_csv = original_csv
        self.generated_csv = generated_csv
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.chexpert_classes = [
            "Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion", 
            "Enlarged Cardiomediastinum", "Lung Opacity", "Pneumonia", "Pneumothorax", 
            "Fracture", "Pleural Other", "Lung Lesion", "Support Devices", "No Finding"
        ]

        self.model = models.inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.model.fc = torch.nn.Identity()  # Use output before classification head
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def extract_features(self, image_paths):
        features = []
        for image_path in tqdm(image_paths, desc="Extracting Features"):
            try:
                image = Image.open(image_path).convert("L")
                image = Image.merge("RGB", (image, image, image))  # Convert grayscale to 3-channel
                image = self.transform(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feature = self.model(image).cpu().numpy().squeeze()
                features.append(feature)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        return np.array(features)

    @staticmethod
    def calculate_fid(features1, features2):
        if features1.shape[0] == 0 or features2.shape[0] == 0:
            return None
        mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
        mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)
        diff = mu1 - mu2
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid

    @staticmethod
    def calculate_kid(features1, features2, subset_size=100, num_subsets=100):
        kid_scores = []
        min_len = min(len(features1), len(features2))
        if min_len < subset_size:
            subset_size = min_len
        for _ in range(num_subsets):
            idx1 = np.random.choice(len(features1), subset_size, replace=False)
            idx2 = np.random.choice(len(features2), subset_size, replace=False)
            feat1, feat2 = features1[idx1], features2[idx2]
            k_xx = polynomial_kernel(feat1, feat1, degree=3, coef0=1)
            k_yy = polynomial_kernel(feat2, feat2, degree=3, coef0=1)
            k_xy = polynomial_kernel(feat1, feat2, degree=3, coef0=1)
            mmd = np.mean(k_xx[np.triu_indices(subset_size, 1)]) + \
                  np.mean(k_yy[np.triu_indices(subset_size, 1)]) - \
                  2 * np.mean(k_xy)
            kid_scores.append(mmd)
        return np.mean(kid_scores), np.std(kid_scores)

    def filter_by_class(self, csv_path, root_path, class_name, limit=None):
        df = pd.read_csv(csv_path)
        df = df[df[class_name] == 1]
        df = df[df['Frontal/Lateral'].str.contains('frontal', case=False, na=False)]
        paths = df['Path'].tolist()
        full_paths = [os.path.join(root_path, p) for p in paths]
        if limit:
            full_paths = full_paths[:limit]
        return full_paths

    def get_all_frontal_paths(self, csv_path, root_path, limit=None):
        df = pd.read_csv(csv_path)
        df = df[df['Frontal/Lateral'].str.contains('frontal', case=False, na=False)]
        paths = df['Path'].tolist()
        full_paths = [os.path.join(root_path, p) for p in paths]
        if limit:
            full_paths = full_paths[:limit]
        return full_paths

    def evaluate_overall(self, limit=14000):
        print("==== Overall Evaluation (All Classes Combined) ====")
        real_paths = self.get_all_frontal_paths(self.original_csv, self.original_root, limit=limit)
        gen_paths = self.get_all_frontal_paths(self.generated_csv, self.generated_root, limit=limit)

        real_features = self.extract_features(real_paths)
        gen_features = self.extract_features(gen_paths)

        fid = self.calculate_fid(real_features, gen_features)
        kid_mean, kid_std = self.calculate_kid(real_features, gen_features)

        print(f"Overall (All Classes) FID: {fid:.2f} | KID: {kid_mean:.4f} ± {kid_std:.4f}")

    def evaluate_per_class(self, limit=1000, min_samples=10):
        print("\n==== Class-wise Evaluation ====")
        for cls in self.chexpert_classes:
            try:
                real_paths = self.filter_by_class(self.original_csv, self.original_root, cls, limit=limit)
                gen_paths = self.filter_by_class(self.generated_csv, self.generated_root, cls, limit=limit)
                if len(real_paths) < min_samples or len(gen_paths) < min_samples:
                    print(f"{cls:22s} | Skipped (too few samples)")
                    continue
                real_features = self.extract_features(real_paths)
                gen_features = self.extract_features(gen_paths)
                fid = self.calculate_fid(real_features, gen_features)
                kid_mean, kid_std = self.calculate_kid(real_features, gen_features)
                print(f"{cls:22s} | FID: {fid:.2f} | KID: {kid_mean:.4f} ± {kid_std:.4f}")
            except Exception as e:
                print(f"{cls:22s} | Error: {e}")

