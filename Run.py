import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader

from Dataset import DashcamDataset
import Model
from Encoder import Encoder
from Decoder import DSA_LSTM
from Train import Trainer
from Train import Trainer_config as config


# dataset
path = "../data/Dashcam_dataset/videos"
yolo_path = "../yolo_features"
resnet_path = "../resnet_features"

train_dataset = DashcamDataset(yolo_path, resnet_path, train=True)
test_dataset = DashcamDataset(yolo_path, resnet_path, train=False)
train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size = config.batch_size, shuffle=True)


# model instance
# encoder = Encoder(config.device).eval()
# encoder = nn.DataParallel(encoder).to(config.device)
lstm = DSA_LSTM(config.device).train().to(config.device)
trainer = Trainer(lstm, train_loader, test_loader, config)

trainer.train()
