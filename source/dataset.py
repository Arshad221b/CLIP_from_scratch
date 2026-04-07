import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import random
import pandas as pd
from collections import defaultdict


class CustomDataset(Dataset):
    def __init__(
        self, data_root="/Users/michelangelo/Coding/CLIP_from_scratch/flickr30k"
    ):
        self.data_root = data_root
        self.image_dir = os.path.join(data_root, "Images")
        captions_file = os.path.join(data_root, "captions.txt")

        # Load captions CSV
        df = pd.read_csv(captions_file)

        # Group captions by image, filtering out null/non-string captions
        self.image_to_captions = defaultdict(list)
        for _, row in df.iterrows():
            caption = row["caption"]
            if pd.notna(caption) and isinstance(caption, str):
                self.image_to_captions[row["image"]].append(caption)

        # Create list of (image_path, captions) tuples, only include images with valid captions
        self.data = [
            (os.path.join(self.image_dir, img), caps)
            for img, caps in self.image_to_captions.items()
            if len(caps) > 0
        ]

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, captions = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        # Randomly select one of the 5 captions
        caption = random.choice(captions)
        return img, caption
