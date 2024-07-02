import torch
import torch.nn as nn

from Model import FeatureExtractor, fasterrcnn

class Encoder(nn.Module):
  def __init__(self, device):
    super(Encoder, self).__init__()
    self.model = fasterrcnn(device)
    self.layer = "roi_heads.box_head.fc7"
    self.features_extractor = FeatureExtractor(self.model, [self.layer]).eval()

  def forward(self, frames):
    batch_size = frames.size(0)
    frames_size = frames.size(1)
    video_features = []

    for i in range(batch_size):
        video = frames[i]

        roi_features = []

        for j in range(frames_size):
          with torch.no_grad():
            frame = video[j, :, :, :].unsqueeze(0)  # (1, 3, 640, 1280)
            feature, _ = self.features_extractor(frame)

            roi_features.append(feature[self.layer])

        roi_features = torch.stack(roi_features, dim=0)  # 모든 frames 하나의 tensor로, [100, 20, 1024]
        video_features.append(roi_features)

    video_features = torch.stack(video_features, dim=0)  # 모든 video 하나의 tensor로, [batch, 100, 20, 1024]

    return video_features
  
  