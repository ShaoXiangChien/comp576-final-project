import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
from utils import *
from model import *
from data_loader import *
from early_stopping import *

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = ImageCaptioningModel(prefix_length=10).to(device)


optimizer = optim.Adam(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler()


num_epochs = 200

val_loss_sim_epoch = []
val_loss_con_epoch = []
val_loss_div_epoch = []
val_loss_total_epoch = []

for epoch in range(num_epochs):
    model.train()
    train_losses = []
    
    # Training phase
    for images, _ in tqdm(train_dataloader, desc=f"Training Epoch {epoch}"):
        images = images.to(device)
        
        with torch.no_grad():
            captions_input = torch.randint(
                low=0, 
                high=model.tokenizer.vocab_size, 
                size=(images.size(0), 10),
                device=device
            )
        
        with torch.cuda.amp.autocast():
            outputs = model(images, input_ids=captions_input) 
            logits = outputs.logits

            captions_text = model.tokenizer.batch_decode(
                torch.argmax(logits, dim=-1),
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )

            clip_inputs = clip_processor(
                text=captions_text,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)

            caption_embeddings = clip_model.get_text_features(
                input_ids=clip_inputs['input_ids'],
                attention_mask=clip_inputs['attention_mask']
            )
            
            denormalized_images = denormalize(images)

            clip_image_inputs = clip_processor(images=denormalized_images, return_tensors="pt").to(device)
            image_embeddings_clip = clip_model.get_image_features(**clip_image_inputs)

            caption_embeddings = nn.functional.normalize(caption_embeddings, p=2, dim=1)
            image_embeddings_clip = nn.functional.normalize(image_embeddings_clip, p=2, dim=1)

            loss_sim = cosine_similarity_loss(image_embeddings_clip, caption_embeddings)
            loss_con = contrastive_loss(image_embeddings_clip, caption_embeddings)
            loss_div = torch.tensor(diversity_loss(torch.argmax(logits, dim=-1), model.tokenizer), device=device)

            alpha, beta, gamma = 1.0, 1.0, 1.0
            loss_total = alpha * loss_sim + beta * loss_con + gamma * loss_div
            train_losses.append(loss_total.item())

        optimizer.zero_grad()
        scaler.scale(loss_total).backward()
        scaler.step(optimizer)
        scaler.update()
    
    model.eval()
    val_loss_sim, val_loss_con, val_loss_div, val_loss_total = 0, 0, 0, 0
    val_count = 0
    
    with torch.no_grad():
        for images, _ in tqdm(val_dataloader, desc=f"Validation Epoch {epoch}"):
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

            clip_inputs = clip_processor(
                text=captions_text,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)

            caption_embeddings = clip_model.get_text_features(
                input_ids=clip_inputs['input_ids'],
                attention_mask=clip_inputs['attention_mask']
            )

            denormalized_images = denormalize(images)

            clip_image_inputs = clip_processor(images=denormalized_images, return_tensors="pt").to(device)
            image_embeddings_clip = clip_model.get_image_features(**clip_image_inputs)
            
            caption_embeddings = nn.functional.normalize(caption_embeddings, p=2, dim=1)
            image_embeddings_clip = nn.functional.normalize(image_embeddings_clip, p=2, dim=1)
            
            loss_sim = cosine_similarity_loss(image_embeddings_clip, caption_embeddings)
            loss_con = contrastive_loss(image_embeddings_clip, caption_embeddings)
            loss_div = torch.tensor(diversity_loss(torch.argmax(logits, dim=-1), model.tokenizer), device=device)
            
            val_loss_sim += loss_sim.item()
            val_loss_con += loss_con.item()
            val_loss_div += loss_div.item()
            val_loss_total += (alpha * loss_sim + beta * loss_con + gamma * loss_div).item()
            val_count += 1

    avg_val_loss_sim = val_loss_sim / val_count
    avg_val_loss_con = val_loss_con / val_count
    avg_val_loss_div = val_loss_div / val_count
    avg_val_loss_total = val_loss_total / val_count

    val_loss_sim_epoch.append(avg_val_loss_sim)
    val_loss_con_epoch.append(avg_val_loss_con)
    val_loss_div_epoch.append(avg_val_loss_div)
    val_loss_total_epoch.append(avg_val_loss_total)

    avg_train_loss = sum(train_losses) / len(train_losses)
    print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss (Sim) = {avg_val_loss_sim:.4f}, Val Loss (Con) = {avg_val_loss_con:.4f}, Val Loss (Div) = {avg_val_loss_div:.4f}, Val Loss (Total) = {avg_val_loss_total:.4f}")
    
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss_sim': avg_val_loss_sim,
            'val_loss_con': avg_val_loss_con,
            'val_loss_div': avg_val_loss_div,
            'val_loss_total': avg_val_loss_total,
        }, f'./config_3_checkpoints/checkpoint_epoch_{epoch + 1}.pt')


val_losses = {
    "val_loss_sim": val_loss_sim_epoch,
    "val_loss_con": val_loss_con_epoch,
    "val_loss_div": val_loss_div_epoch,
    "val_loss_total": val_loss_total_epoch,
}
with open('./config_3_checkpoints/val_losses.json', 'w') as f:
    json.dump(val_losses, f)
