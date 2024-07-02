import torch
import torchvision
import pickle
from torch.utils.data import Dataset
import os

# params
BATCH_SIZE = 16


# # Dataset
# class DashcamDataset(Dataset):
#   def __init__(self, data_path, train=True, transform=None):
#     self.data_path = data_path
#     self.transform = transform

#     if train == True:
#       self.base_path = os.path.join(data_path, "training")
#     else:
#       self.base_path = os.path.join(data_path, "testing")

#     self.positive_path = os.path.join(self.base_path, "positive")
#     self.negative_path = os.path.join(self.base_path, "negative")

#     self.positive_videos = [os.path.join(self.positive_path, v) for v in sorted(os.listdir(self.positive_path))]
#     self.negative_videos = [os.path.join(self.negative_path, v) for v in sorted(os.listdir(self.negative_path))]

#     self.video_paths = self.positive_videos + self.negative_videos

#   def __len__(self):
#     return len(self.video_paths)

#   def __getitem__(self, idx):
#     video_path = self.video_paths[idx]
#     video = torchvision.io.read_video(video_path, output_format = 'TCHW')[0]
#     video = torch.stack([resize(frame, (180, 320)) for frame in video])  # Resize
#     label = 1 if video_path in self.positive_videos else 0

#     return video, label



# Dataset
class DashcamDataset(Dataset):
  def __init__(self, yolo_path, resnet_path, train=False):
    self.yolo_path = yolo_path
    self.resnet_path = resnet_path
    self.train = train

    yolo_path = self.get_path(self.yolo_path)
    resnet_path = self.get_path(self.resnet_path)

    self.yolo_positive_videos = [os.path.join(yolo_path[0], v) for v in sorted(os.listdir(yolo_path[0]))]
    self.yolo_negative_videos = [os.path.join(yolo_path[1], v) for v in sorted(os.listdir(yolo_path[1]))]
    self.resnet_positive_videos = [os.path.join(resnet_path[0], v) for v in sorted(os.listdir(resnet_path[0]))]
    self.resnet_negative_videos = [os.path.join(resnet_path[1], v) for v in sorted(os.listdir(resnet_path[1]))]
        
    self.yolo_video_paths = self.yolo_positive_videos + self.yolo_negative_videos
    self.resnet_video_paths = self.resnet_positive_videos + self.resnet_negative_videos
  
  def get_path(self, path):
    if self.train == True:
      base_path = "training"
    else:
      base_path = "testing"

    positive_path = os.path.join(base_path, "positive")
    negative_path = os.path.join(base_path, "negative")
    
    return [os.path.join(path, positive_path), os.path.join(path, negative_path)]
    
    
  def __len__(self):
    return len(self.resnet_video_paths)

  def __getitem__(self, idx):
    # yolo = self.yolo_video_paths[idx]
    resnet = self.resnet_video_paths[idx]
    
    with open(resnet, 'rb') as f:
      resnet_features = pickle.load(f)
    
    features = torch.tensor(resnet_features)
    label = 1 if resnet in self.resnet_positive_videos else 0

    return features, label


