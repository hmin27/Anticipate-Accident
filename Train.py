import torch
import torch.nn as nn
import os
from tqdm import tqdm
import wandb

import Model
from Loss import AccidentLoss, Loss


class Trainer_config():
    # optimization parameters
    max_epochs = 50
    batch_size = 16
    learning_rate = 1e-4
    betas = (0.9, 0.95)
    weight_decay = 0.1 
    lr_decay = False
    
    # checkpoint settings\
    ckpt_path = 'ckpt_50'
    num_workers = 0 # for DataLoader
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


class Trainer():
  def __init__(self, decoder, train_loader, test_loader, config):
    # self.encoder = encoder
    self.decoder = decoder
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.config = config
    # self.loss = Loss()
    self.loss = AccidentLoss(n_frames=100, device=config.device)
    
    wandb.init(project='dashcam-review')


  def save_checkpoint(self, epoch):
    ckpt_filename = f'epoch_{epoch}.ckpt'
    ckpt_path = os.path.join(self.config.ckpt_path, ckpt_filename)
    
    os.makedirs(self.config.ckpt_path, exist_ok=True)
    torch.save(self.decoder.state_dict(), ckpt_path)
    
  def train(self):
    optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.config.learning_rate)

    for epoch in range(self.config.max_epochs):
      print(f"Epoch: {epoch+1}")

      train_loss = 0
      step = 0

      pbar = tqdm(self.test_loader, total=len(self.test_loader))
      for features, labels in pbar:
        step += 1
        optimizer.zero_grad()

        # prediction
        features = features.float().to(self.config.device)
        
        # augment label to probability
        labels = labels.to(self.config.device)
        aug_labels = torch.stack([torch.tensor([0.0]*100) if label==0 else torch.tensor([1.0]*100) for label in labels])

        # features = self.encoder(frames)
        preds = self.decoder(features).squeeze(-1)  # [batch_size, frames=100]

        # Loss
        total_loss = 0

        # for i in range(preds.size(0)):
        #   pred = self.loss.prediction(preds[i])

          # if labels[i] == 1:
          #   pos_loss = self.loss.AnticipationLoss(pred)
          #   total_loss += pos_loss

          # else:
          #   neg_loss = self.loss.CrossEntropyLoss(pred)
          #   total_loss += neg_loss
        
        total_loss += self.loss(preds.permute(1, 0, 2), aug_labels.permute(1,0).to(self.config.device))

        # Train loss
        total_loss = total_loss / preds.size(0)
        train_loss += total_loss.item()  # Tensor to Scalar

        # back propagation
        total_loss.backward()
        optimizer.step()
        
        wandb.log({'Train Loss': total_loss.item()})

        pbar.set_description(f"Epoch {epoch+1}/{self.config.max_epochs}, Train Loss: {total_loss.item():.4f}", refresh=True)

      avg_train_loss = train_loss / step

      print(f"Average Train Loss: {avg_train_loss:.4f}")

      if epoch+1 % 10 == 0:
        self.save_checkpoint(epoch+1)