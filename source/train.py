import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from tqdm import tqdm

from encoders import CLIPModel
from cliploss import CLIPLoss
from dataset import CustomDataset


def get_device():
    """Get best available device for MacBook."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_clip(
    num_epochs=10,
    batch_size=128,  
    embed_dim=256,
    learning_rate=1e-4,
    data_root="/Users/michelangelo/Coding/CLIP_from_scratch/flickr30k",
):
    device = get_device()
    print(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Load model
    model = CLIPModel(embed_dim=embed_dim).to(device)
    loss_fn = CLIPLoss(temperature=0.07)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )

    # Load dataset
    print("Loading dataset...")
    dataset = CustomDataset(data_root=data_root)

    # Custom collate function to handle text tokenization
    def collate_fn(batch):
        images, captions = zip(*batch)
        images = torch.stack(images)

        # Tokenize captions
        encoded = tokenizer(
            list(captions),
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )

        return images, encoded["input_ids"], encoded["attention_mask"]

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Avoid multiprocessing issues on Mac
    )

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for images, input_ids, attention_mask in pbar:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Forward pass
            image_features, text_features = model(images, input_ids, attention_mask)

            # Compute loss
            loss = loss_fn.compute_loss(image_features, text_features)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            },
            f"clip_checkpoint_epoch_{epoch + 1}.pt",
        )

    print("Training complete!")
    return model


if __name__ == "__main__":
    train_clip()
