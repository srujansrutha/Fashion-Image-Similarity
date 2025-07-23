import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

class CLIPFashionTrainer:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.train()

    def train_epoch(self, loader, optimizer):
        total, count = 0,0
        bar = tqdm(loader)
        for batch in bar:
            optimizer.zero_grad()
            pv = batch['pixel_values'].to(self.device)
            ids = batch['input_ids'].to(self.device)
            mask= batch['attention_mask'].to(self.device)
            outputs = self.model(pixel_values=pv, input_ids=ids, attention_mask=mask, return_loss=True)
            loss = outputs.loss
            loss.backward(); optimizer.step()
            total += loss.item(); count +=1
            bar.set_postfix(loss=total/count)
        return total/count

    def train(self, dataset, epochs=5, bs=8, lr=1e-5):
        dl = DataLoader(dataset, batch_size=bs, shuffle=True)
        opt = optim.AdamW(self.model.parameters(), lr=lr)
        for e in range(epochs):
            avg = self.train_epoch(dl, opt)
            torch.save(self.model.state_dict(), f'clip_epoch{e+1}.pt')
        torch.save(self.model.state_dict(), 'clip_fashion_final.pt')
