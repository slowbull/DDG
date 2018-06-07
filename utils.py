import os, sys, time
import numpy as np
import random
import torch
from torch.autograd import Variable

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, gammas, schedule, lr):
  """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
  assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
  for (gamma, step) in zip(gammas, schedule):
    if (epoch >= step):
      lr = lr * gamma
    else:
      break
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  return lr


def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0 / batch_size))
  return res


def compute_shapes(models, args):
  shapes = []
  # switch to evaluate mode
  for model in models:
    model.eval()

  inputs = torch.FloatTensor(1, 3, 32, 32)
  inputs_var = Variable(inputs)

  output = models[0](inputs_var)
  size = output.size()
  shape = [args.batch_size, ] + [x for x in size[1:]]
  shapes.append(shape)

  for model, gpu in zip(models[1:(args.splits-1)], args.dist_gpus[1:(args.splits-1)]):
    output = model(output)
    size = output.size()
    shape = [args.batch_size, ] + [x for x in size[1:]]
    shapes.append(shape)

  return shapes

