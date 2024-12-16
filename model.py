import torch.nn as nn
from torchvision import models
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class ImageEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.mobilenet.classifier = nn.Identity()

    def forward(self, images):
        return self.mobilenet(images)

class ImagePrefixMapper(nn.Module):
    """
    Maps image embeddings into a sequence of prefix embeddings that the decoder can use.
    """
    def __init__(self, image_emb_dim, prefix_length, decoder_emb_dim):
        super().__init__()
        self.prefix_length = prefix_length
        self.mlp = nn.Sequential(
            nn.Linear(image_emb_dim, 512),
            nn.ReLU(),
            nn.Linear(512, prefix_length * decoder_emb_dim)
        )

    def forward(self, image_emb):
        proj_out = self.mlp(image_emb)
        return proj_out.view(proj_out.size(0), self.prefix_length, -1)
    
class ImageCaptioningModel(nn.Module):
    def __init__(self, prefix_length=10):
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=True)
        self.decoder = AutoModelForCausalLM.from_pretrained('distilgpt2')
        self.tokenizer = AutoTokenizer.from_pretrained('distilgpt2')

        self.tokenizer.padding_side = 'left'

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token

        decoder_emb_dim = self.decoder.transformer.wte.weight.shape[1]

        image_emb_dim = 1280
        self.prefix_mapper = ImagePrefixMapper(image_emb_dim, prefix_length, decoder_emb_dim)

        self.prefix_length = prefix_length

    def forward(self, images, input_ids, attention_mask=None, labels=None):
        image_emb = self.image_encoder(images)
        prefix_embs = self.prefix_mapper(image_emb)

        input_embs = self.decoder.transformer.wte(input_ids)

        full_embs = torch.cat([prefix_embs, input_embs], dim=1)

        batch_size, seq_len = input_ids.size()
        full_seq_len = seq_len + self.prefix_length
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len).to(input_ids.device)
        prefix_mask = torch.ones(batch_size, self.prefix_length).to(input_ids.device)
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        
        def replace_forward(module, input, output):
            return full_embs

        handle = self.decoder.transformer.wte.register_forward_hook(replace_forward)

        outputs = self.decoder(
            input_ids=torch.zeros(batch_size, full_seq_len, dtype=torch.long, device=input_ids.device),
            attention_mask=full_attention_mask,
            labels=labels,
        )

        handle.remove()

        return outputs

   
    @torch.no_grad()
    def generate(self, images, max_length=20):
        device = next(self.parameters()).device
        images = images.to(device)
        
        image_emb = self.image_encoder(images)
        prefix_embs = self.prefix_mapper(image_emb)
        
        batch_size = images.size(0)
        generated_sequences = []
        
        for i in range(batch_size):
            single_prefix = prefix_embs[i:i+1]
            
            current_token_ids = torch.tensor([[self.tokenizer.bos_token_id]]).to(device)
            generated_tokens = []
            
            for _ in range(max_length):
                def replace_forward(module, input, output):
                    current_token_emb = output[:, -1:, :]
                    if len(generated_tokens) == 0:
                        return torch.cat([single_prefix, current_token_emb], dim=1)
                    else:
                        return current_token_emb
                
                handle = self.decoder.transformer.wte.register_forward_hook(replace_forward)
                
                try:
                    outputs = self.decoder(
                        input_ids=current_token_ids,
                        use_cache=False
                    )
                finally:
                    handle.remove()
                
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()
                
                generated_tokens.append(next_token_id)
                
                current_token_ids = torch.tensor([[next_token_id]]).to(device)
                
                if next_token_id == self.tokenizer.eos_token_id:
                    break
            
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_sequences.append(generated_text)
        
        return generated_sequences
        