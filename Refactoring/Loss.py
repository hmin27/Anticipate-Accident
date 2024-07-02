# a_0 : Probability of Accident
# a_1 : Probability of Non-Accident
# p(prob, frame)
# t(prob, frame)

import torch
import torch.nn.functional as F


class Loss():
  def __init__(self):
    pass

  def prediction(self, pred):  # [100]
    stacked_pred = torch.stack([torch.tensor([p, frame]) for frame, p in enumerate(pred)])

    return stacked_pred


  def CrossEntropyLoss(self, pred):  # for negative
    label = torch.stack([p[0] for p in pred])
    loss = -torch.sum(torch.log(1-label))
    loss.requires_grad = True

    return loss


  def AnticipationLoss(self, pred):  # for positive
    label = torch.stack([p[0] for p in pred])  # pred prob
    timestep = torch.stack([p[1] for p in pred]).to(int)
    loss = -torch.sum(torch.exp(-torch.maximum(torch.tensor(0), (90 - timestep)))*torch.log(label))
    loss.requires_grad = True

    return loss
  
  
class AccidentLoss(object):
  def __init__(self, n_frames, device):
      self.n_frames = n_frames
      pos_weights = torch.exp(
          - torch.arange(self.n_frames - 1, -1, -1) / 20.0).view(-1, 1)
      neg_weights = torch.ones((n_frames, 1))
      # (n_frames x 2)
      self.frame_weights = torch.cat([neg_weights, pos_weights], dim=1)
      self.frame_weights = self.frame_weights.to(device)
      self.frame_weights.requires_grad = False
      self.log_softmax = torch.nn.LogSoftmax(dim=-1)
      self.nll_loss = torch.nn.NLLLoss()
  
  def __call__(self, logits, labels):
      # (n_frames x B x 2)
      loss = self.log_softmax(logits)
      # (n_frames x B x 2) multiply each frame's outputs with specific weight
      loss = torch.mul(self.frame_weights.unsqueeze(-2), loss)
      # (n_frames*B x 2) following NLLLoss's expected input of (minibatch, C)
      loss = loss.contiguous().view(-1, 2)
      labels = labels.contiguous().view(-1)
      # compute average loss over all frames of entire batch
      loss = self.nll_loss(loss, labels.type(torch.LongTensor).to('cuda'))
      return loss