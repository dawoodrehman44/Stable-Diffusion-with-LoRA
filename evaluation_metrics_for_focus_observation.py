import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from captum.attr import LayerGradCam
from scipy.stats import pearsonr

class CXRFocusMetrics:
    def __init__(self, model_weights_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_config = {
            "load_backbone_weights": "checkpoints/cxrclip_mc/r50_mc.pt",
            "freeze_backbone_weights": False,
            "image_encoder": {
                "name": "resnet",
                "resnet_type": "resnet50",
                "pretrained": True,
                "source": "cxr_clip"
            },
            "classifier": {
                "config": {
                    "name": "linear",
                    "n_class": 14
                }
            }
        }

        self.model = CXRClassification(model_config=model_config, model_type="resnet")
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        target_layer = self.model.image_encoder.resnet.layer4[-1]
        self.gradcam = LayerGradCam(self._forward_wrapper, target_layer)

    def _forward_wrapper(self, x):
        dummy_labels = torch.zeros(x.size(0), dtype=torch.long).to(self.device)
        output = self.model({"images": x.to(self.device), "labels": dummy_labels})
        return output["cls_pred"] if isinstance(output, dict) else output

    def load_image(self, path):
        image = Image.open(path).convert("RGB")
        return self.transform(image).unsqueeze(0)

    def compute_focus_metrics(self, image_paths, target_class, sparsity_threshold=0.85):
        confidences = []
        entropies = []
        sparsities = []

        for path in image_paths:
            image = self.load_image(path).to(self.device)

            with torch.no_grad():
                output = self.model({"images": image, "labels": torch.zeros(1, dtype=torch.long).to(self.device)})
                prob = torch.sigmoid(output["cls_pred"]).squeeze()[target_class].item()
                confidences.append(prob)

                cam = self.gradcam.attribute(image, target=target_class).squeeze().detach().cpu().numpy()
                cam = np.maximum(cam, 0)
                cam = cam / (cam.sum() + 1e-8)

            entropy = -np.sum(cam * np.log(cam + 1e-8))
            entropies.append(entropy)

            flat_cam = cam.flatten()
            sorted_cam = np.sort(flat_cam)[::-1]
            cumsum = np.cumsum(sorted_cam)
            k = np.searchsorted(cumsum, sparsity_threshold)
            sparsity = k / len(flat_cam)
            sparsities.append(sparsity)

        conf_focus_corr = pearsonr(confidences, entropies)[0]
        confidence_var = np.var(confidences)
        mean_entropy = np.mean(entropies)
        mean_sparsity = np.mean(sparsities)

        return {
            "Confidence–Focus Correlation": conf_focus_corr,
            "Confidence Variance": confidence_var,
            "Average Focus Entropy": mean_entropy,
            "Average Focus Sparsity": mean_sparsity
        }


# Example usage (commented):
# if __name__ == "__main__":
#     cxr_metrics = CXRFocusMetrics(model_weights_path="/mnt/Internal/MedImage/CheXpert Dataset/Lab_Rotation_2/cxr_clip_training/model_epoch_10.pth")
#     image_paths = [
#         "/mnt/Internal/MedImage/unzip_chexpert_images/CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg",
#         "/mnt/Internal/MedImage/unzip_chexpert_images/CheXpert-v1.0/train/patient00002/study1/view1_frontal.jpg",
#         "/mnt/Internal/MedImage/unzip_chexpert_images/CheXpert-v1.0/train/patient00003/study1/view1_frontal.jpg"
#     ]
#     target_class = 12  # Cardiomegaly
#     results = cxr_metrics.compute_focus_metrics(image_paths, target_class)
#     for k, v in results.items():
#         print(f"{k}: {v:.4f}")
