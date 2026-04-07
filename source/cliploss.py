import torch
import torch.nn.functional as F


class CLIPLoss:
    def __init__(self, temperature=0.07):
        self.temperature = temperature

    def compute_loss(self, img_features, text_features):
        # Normalize the features
        img_features = F.normalize(img_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # Compute cosine similarity
        logits = torch.matmul(img_features, text_features.T) / self.temperature

        # Create labels for contrastive loss
        batch_size = img_features.size(0)
        labels = torch.arange(batch_size).to(img_features.device)

        # Compute cross-entropy loss
        loss_img_to_text = F.cross_entropy(logits, labels)
        loss_text_to_img = F.cross_entropy(logits.T, labels)

        return (loss_img_to_text + loss_text_to_img) / 2
