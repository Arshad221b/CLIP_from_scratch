# CLIP from Scratch

A from-scratch PyTorch implementation of [CLIP (Contrastive Language–Image Pretraining)](https://arxiv.org/abs/2103.00020), trained on the Flickr30k dataset and optimized for local training on Apple Silicon (MPS).

---

## Overview

CLIP learns a joint embedding space for images and text using contrastive learning. Matching image–caption pairs are pulled together while non-matching pairs are pushed apart. At inference time, the model can retrieve the most relevant image for a given text query, or vice versa.

This implementation uses lightweight backbone models to make training feasible on a MacBook.

---

## Architecture

### Image Encoder
- **Backbone:** MobileNetV3-Small (pretrained on ImageNet)
- **Pooling:** Adaptive average pooling → flatten
- **Projection:** Linear(576 → 512) → ReLU → Linear(512 → 256)

### Text Encoder
- **Backbone:** DistilBERT (`distilbert-base-uncased`, pretrained)
- **Representation:** `[CLS]` token from last hidden state
- **Projection:** Linear(768 → 256)

### Shared Embedding Space
- Both encoders project to a 256-dimensional embedding space
- A learnable temperature parameter (initialized to 0.07) scales the logits

---

## Loss Function

Symmetric contrastive loss (InfoNCE):

```
loss = (CE(image→text) + CE(text→image)) / 2
```

Both image and text features are L2-normalized before computing cosine similarity. Labels are diagonal (each image matches its own caption).

---

## Dataset

**Flickr30k** — 31,000 images with 5 captions each (~155,000 image–text pairs).

Expected directory structure:
```
flickr30k/
├── Images/
│   ├── 1000092795.jpg
│   └── ...
└── captions.txt
```

`captions.txt` should be a CSV with `image` and `caption` columns. The dataset class filters out null/non-string captions and randomly samples one of the 5 captions per image per training step.

---

## Training

### Hyperparameters

| Parameter       | Value              |
|-----------------|--------------------|
| Embedding dim   | 256                |
| Temperature     | 0.07 (learnable)   |
| Batch size      | 128                |
| Epochs          | 10                 |
| Optimizer       | AdamW              |
| Learning rate   | 1e-4               |
| Weight decay    | 0.01               |
| Max token length| 77                 |

### Device Support

Training automatically uses Apple Silicon MPS if available, falling back to CPU.

### Run Training

```bash
cd source
python train.py
```

Checkpoints are saved after each epoch as `clip_checkpoint_epoch_{N}.pt`.

---

## Evaluation

The evaluation script measures cross-modal retrieval performance on a random sample of the dataset using Recall@K metrics.

```bash
cd source
python evaluate.py
```

Metrics reported:
- **Image-to-Text Retrieval:** Recall@1, @5, @10
- **Text-to-Image Retrieval:** Recall@1, @5, @10

The script also prints sample query/retrieval pairs to give a qualitative sense of model behavior.

By default it loads `clip_checkpoint_epoch_10.pt` and evaluates on 1,000 random samples.

---

## Project Structure

```
.
├── source/
│   ├── encoders.py        # ImageEncoder, TextEncoder, CLIPModel
│   ├── cliploss.py        # Symmetric contrastive loss
│   ├── dataset.py         # Flickr30k dataset loader
│   ├── train.py           # Training loop with checkpointing
│   ├── evaluate.py        # Recall@K retrieval evaluation
│   └── check_dataset.py   # Dataset inspection utility
├── flickr30k/             # Dataset (not tracked in git)
└── README.md
```

---

## Dependencies

```
torch
torchvision
transformers
Pillow
pandas
tqdm
numpy
```

Install with:

```bash
pip install torch torchvision transformers pillow pandas tqdm numpy
```

---

## References

- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (OpenAI CLIP, 2021)
- [Flickr30k Dataset](https://shannon.cs.illinois.edu/DenotationGraph/)
- [DistilBERT](https://arxiv.org/abs/1910.01108)
- [MobileNetV3](https://arxiv.org/abs/1905.02244)
