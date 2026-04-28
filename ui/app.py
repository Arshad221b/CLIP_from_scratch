import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'source'))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import DistilBertTokenizer
import gradio as gr
import numpy as np

from encoders import CLIPModel

CHECKPOINT = os.path.join(os.path.dirname(__file__), '..', 'source', 'clip_checkpoint_epoch_10.pt')
EMBED_DIM = 256
DEVICE = 'cpu'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

model = CLIPModel(embed_dim=EMBED_DIM)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
state = ckpt.get('model_state_dict', ckpt)
model.load_state_dict(state)
model.eval()

# In-memory image store: list of (pil_image, embedding)
store = []

def encode_image(pil_img):
    t = transform(pil_img.convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        emb = model.image_encoder(t)
    return F.normalize(emb, dim=-1).squeeze(0)

def encode_text(text):
    enc = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        emb = model.text_encoder(enc['input_ids'], enc['attention_mask'])
    return F.normalize(emb, dim=-1).squeeze(0)

def upload_images(files):
    if not files:
        return "No files provided."
    added = 0
    for f in files:
        try:
            path = f.name if hasattr(f, 'name') else f
            img = Image.open(path).convert('RGB')
            emb = encode_image(img)
            store.append((img, emb))
            added += 1
        except Exception as e:
            pass
    return f"Uploaded {added} image(s). Total in index: {len(store)}"

def search(query, top_k=5):
    if not store:
        return [], "No images in index. Upload some first."
    if not query.strip():
        return [], "Enter a search query."
    q_emb = encode_text(query)
    embs = torch.stack([e for _, e in store])
    sims = (embs @ q_emb).cpu().numpy()
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = []
    for i in top_idx:
        img, _ = store[i]
        results.append((img, f"{sims[i]:.3f}"))
    return results, f"Top {len(results)} results for: '{query}'"

with gr.Blocks(title="CLIP Search") as demo:
    gr.Markdown("## CLIP Image Search")

    with gr.Row():
        upload = gr.File(label="Upload Images", file_count="multiple", file_types=["image"])
        upload_btn = gr.Button("Index Images")
    upload_status = gr.Textbox(label="Status", interactive=False)
    upload_btn.click(upload_images, inputs=upload, outputs=upload_status)

    gr.Markdown("---")
    with gr.Row():
        query_box = gr.Textbox(label="Search query", placeholder="a dog running on grass")
        search_btn = gr.Button("Search")
    search_status = gr.Textbox(label="", interactive=False)
    gallery = gr.Gallery(label="Results", columns=5, height=300)
    search_btn.click(search, inputs=query_box, outputs=[gallery, search_status])
    query_box.submit(search, inputs=query_box, outputs=[gallery, search_status])

if __name__ == '__main__':
    demo.launch()
