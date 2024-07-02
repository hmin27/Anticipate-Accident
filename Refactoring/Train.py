import torch
import torch.nn as nn
import os
from tqdm import tqdm
import wandb

import Model
from Loss import Loss


class Trainer_config():
    # optimization parameters
    max_epochs = 10
    batch_size = 16
    learning_rate = 1e-3
    betas = (0.9, 0.99)
    weight_decay = 0.1 
    lr_decay = True
    
    # checkpoint settings\
    ckpt_path = 'ckpt.pth'
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
    self.loss = Loss()
    
    wandb.init(project='Ai_Hanmuncheol')


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
        labels = labels.to(self.config.device)

        # features = self.encoder(frames)
        preds = self.decoder(features).squeeze(-1)  # [batch_size, frames=100]
        print(preds.shape)

        # Loss
        total_loss = 0

        for i in range(preds.size(0)):
          pred = self.loss.prediction(preds[i])
          # print(pred)

          if labels[i] == 1:
            pos_loss = self.loss.AnticipationLoss(pred)
            total_loss += pos_loss

          else:
            neg_loss = self.loss.CrossEntropyLoss(pred)
            total_loss += neg_loss

        # Train loss
        total_loss = total_loss / preds.size(0)
        train_loss += total_loss.item()  # Tensor to Scalar

        # back propagation
        total_loss.backward()
        optimizer.step()
        
        wandb.log({'Train Loss': total_loss.item()})

        pbar.set_description(f"Epoch {epoch+1}/{self.config.max_epochs}, Train Loss: {total_loss.item():.4f}", refresh=True)

        # if step % 10 == 0:
        #   print(f"Step {step}/{len(self.test_loader)}, Train Loss: {total_loss.item():.4f}")
        #   self.save_checkpoint()

      avg_train_loss = train_loss / step

      # train_loss_save.append(avg_train_loss)
      print(f"Average Train Loss: {avg_train_loss:.4f}")

      self.save_checkpoint(epoch+1)