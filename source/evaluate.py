import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from tqdm import tqdm
import random
import numpy as np

from encoders import CLIPModel
from dataset import CustomDataset


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_retrieval(
    checkpoint_path="clip_checkpoint_epoch_10.pt",
    data_root="/Users/michelangelo/Coding/CLIP_from_scratch/flickr30k",
    batch_size=64,
    k_values=[1, 5, 10],
    num_test_samples=1000,
):
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Load model
    model = CLIPModel(embed_dim=256).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load dataset
    print("Loading dataset...")
    dataset = CustomDataset(data_root=data_root)

    # Sample test indices
    test_indices = random.sample(
        range(len(dataset)), min(num_test_samples, len(dataset))
    )

    print(f"\nEvaluating on {len(test_indices)} samples...")

    # Encode all images and texts
    all_image_features = []
    all_text_features = []
    all_captions = []

    def collate_fn(batch):
        images, captions = zip(*batch)
        images = torch.stack(images)
        encoded = tokenizer(
            list(captions),
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        return images, encoded["input_ids"], encoded["attention_mask"], list(captions)

    # Process in batches
    for i in tqdm(range(0, len(test_indices), batch_size), desc="Encoding"):
        batch_indices = test_indices[i : i + batch_size]
        batch_data = [dataset[idx] for idx in batch_indices]
        images, input_ids, attention_mask, captions = collate_fn(batch_data)

        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            image_features, text_features = model(images, input_ids, attention_mask)

        all_image_features.append(image_features.cpu())
        all_text_features.append(text_features.cpu())
        all_captions.extend(captions)

    # Concatenate all features
    all_image_features = torch.cat(all_image_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)

    # Normalize for cosine similarity
    all_image_features = nn.functional.normalize(all_image_features, dim=1)
    all_text_features = nn.functional.normalize(all_text_features, dim=1)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(all_image_features, all_text_features.T)

    print(f"\n{'=' * 50}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 50}")

    # Image-to-Text Retrieval
    print("\nImage-to-Text Retrieval:")
    print("-" * 30)

    for k in k_values:
        correct = 0
        for i in range(len(test_indices)):
            # Get top-k text matches for image i
            scores = similarity_matrix[i]
            top_k_indices = torch.topk(scores, k).indices

            # In Flickr30k, each image has 5 captions that are consecutive
            # We need to check if any of the correct captions are in top-k
            # For simplicity, we'll consider it correct if the query's own caption is in top-k
            if i in top_k_indices:
                correct += 1

        recall = correct / len(test_indices) * 100
        print(f"  Recall@{k}: {recall:.2f}%")

    # Text-to-Image Retrieval
    print("\nText-to-Image Retrieval:")
    print("-" * 30)

    for k in k_values:
        correct = 0
        for i in range(len(test_indices)):
            # Get top-k image matches for text i
            scores = similarity_matrix[:, i]
            top_k_indices = torch.topk(scores, k).indices

            if i in top_k_indices:
                correct += 1

        recall = correct / len(test_indices) * 100
        print(f"  Recall@{k}: {recall:.2f}%")

    # Show some random examples
    print(f"\n{'=' * 50}")
    print("SAMPLE MATCHES")
    print(f"{'=' * 50}")

    num_examples = 3
    for _ in range(num_examples):
        idx = random.randint(0, len(test_indices) - 1)

        # Get top match
        scores = similarity_matrix[idx]
        top_idx = torch.argmax(scores).item()

        print(f"\nQuery (Image {test_indices[idx]}):")
        print(f"  True caption: {all_captions[idx][:100]}...")
        print(f"  Retrieved caption: {all_captions[top_idx][:100]}...")
        print(f"  Match {'CORRECT' if idx == top_idx else 'INCORRECT'}")

    print(f"\n{'=' * 50}")


if __name__ == "__main__":
    evaluate_retrieval()
