import torch
import torchvision
import torch.nn as nn
from typing import Dict, Iterable, Callable

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


def fasterrcnn(device):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    fasterrcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights,
                                                                    rpn_post_nms_top_n_train=20,
                                                                    rpn_post_nms_top_n_test=20,
                                                                    box_detections_per_img=20)
    fasterrcnn = fasterrcnn.eval().to(device)

    return fasterrcnn


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x):
        result = self.model(x)
        return self._features, result


class Attention(nn.Module):
  def __init__(self, encoder_dim):
    super(Attention, self).__init__()
    self.U = nn.Linear(encoder_dim, 512)
    self.W = nn.Linear(encoder_dim, 512)
    self.w = nn.Linear(512, 1)
    self.tanh = nn.Tanh()
    self.softmax = nn.Softmax(-1)

  def forward(self, img_features, hidden_state):
    W_e = self.W(img_features)  # (batch, 100, 20, 512)
    U_e = self.U(hidden_state).unsqueeze(0).permute(2, 0, 1, 3)  # (batch, 1, 1, 512)
    att = self.tanh(W_e + U_e)  # (batch, 100, 20, 512)
    e = self.w(att).squeeze(3)  # (batch, 100, 20)
    alpha = self.softmax(e)  # (batch, 100, 20)
    phi = (img_features * alpha.unsqueeze(3)).sum(2)  # phi(x, alpha), (batch, 100, 1000)
    return phi


