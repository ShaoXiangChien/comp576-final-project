import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils import *
from model import *
from data_loader import *

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = ImageCaptioningModel(prefix_length=10).to(device)
checkpoint = torch.load('./config_3_checkpoints/checkpoint_epoch_190.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def compute_alignment_score(dataloader, clip_model, clip_processor, model, device, num_samples=500):
    cosine_similarities = []
    count = 0
    model.eval()
    clip_model.eval()
    
    with torch.no_grad():
        for images, img_paths in dataloader:
            images = images.to(device)

            captions_input = torch.randint(
                low=0, 
                high=model.tokenizer.vocab_size, 
                size=(images.size(0), 10),
                device=device
            )

            outputs = model(images, input_ids=captions_input)
            logits = outputs.logits
            captions_text = model.tokenizer.batch_decode(
                torch.argmax(logits, dim=-1),
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )

            denormalized_images = denormalize(images)

            clip_inputs = clip_processor(images=denormalized_images, return_tensors="pt").to(device)

          
            clip_text_inputs = clip_processor(
                text=captions_text, 
                return_tensors="pt", 
                padding=True
            ).to(device)

            image_embeddings = clip_model.get_image_features(**clip_inputs)
            text_embeddings = clip_model.get_text_features(**clip_text_inputs)

            image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

            cos_sim = (image_embeddings * text_embeddings).sum(dim=1)
            cosine_similarities.extend(cos_sim.cpu().tolist())

            count += images.size(0)
            if count >= num_samples:
                break

    return sum(cosine_similarities) / len(cosine_similarities)


def compute_baseline_score(dataloader, clip_model, clip_processor, device, baseline_caption="A photo of something", num_samples=500):
    cosine_similarities = []
    count = 0
    clip_model.eval()
    
    with torch.no_grad():
        for images, img_paths in dataloader:
            images = images.to(device)
            batch_size = images.size(0)
            baseline_captions = [baseline_caption] * batch_size


            denormalized_images = denormalize(images)

            clip_inputs = clip_processor(images=denormalized_images, return_tensors="pt").to(device)

            
            clip_text_inputs = clip_processor(
                text=baseline_captions, 
                return_tensors="pt", 
                padding=True
            ).to(device)

            image_embeddings = clip_model.get_image_features(**clip_inputs)
            text_embeddings = clip_model.get_text_features(**clip_text_inputs)

            image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

            cos_sim = (image_embeddings * text_embeddings).sum(dim=1)
            cosine_similarities.extend(cos_sim.cpu().tolist())

            count += batch_size
            if count >= num_samples:
                break

    return sum(cosine_similarities) / len(cosine_similarities)


alignment_score = compute_alignment_score(val_dataloader, clip_model, clip_processor, model, device, num_samples=500)
baseline_score = compute_baseline_score(val_dataloader, clip_model, clip_processor, device, baseline_caption="A photo of something", num_samples=500)

print(f"Model Alignment Score: {alignment_score:.4f}")
print(f"Baseline Alignment Score: {baseline_score:.4f}")