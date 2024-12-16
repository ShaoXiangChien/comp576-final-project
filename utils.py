import torch
import torch.nn as nn
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from transformers import CLIPProcessor, CLIPModel

cosine_similarity = nn.CosineSimilarity(dim=1)
device = "mps" if torch.backends.mps.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def cosine_similarity_loss(image_embeddings, caption_embeddings):
    cos_sim = cosine_similarity(image_embeddings, caption_embeddings)
    loss = 1 - cos_sim.mean()
    return loss

def contrastive_loss(image_embeddings, caption_embeddings, temperature=0.07):
    batch_size = image_embeddings.size(0)
    labels = torch.arange(batch_size)
    logits = torch.matmul(image_embeddings, caption_embeddings.T) / temperature
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss

def diversity_loss(captions_ids_batch, tokenizer):
    captions_text = [tokenizer.decode(c, skip_special_tokens=True) for c in captions_ids_batch]
    bleu_scores = []
    
    for i, caption in enumerate(captions_text):
        hypothesis_tokens = caption.strip().split()
        references = captions_text[:i] + captions_text[i+1:]
        references_tokens = [r.strip().split() for r in references]

        if len(references_tokens) == 0 or len(hypothesis_tokens) == 0:
            bleu_scores.append(0.0)
            continue

        score = sentence_bleu(references_tokens, hypothesis_tokens)
        bleu_scores.append(score)

    return np.mean(bleu_scores) if len(bleu_scores) > 0 else 0.0

def denormalize(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1) 
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
    return image_tensor * std + mean