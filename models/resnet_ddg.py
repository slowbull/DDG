import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

from collections import deque


class DownsampleA(nn.Module):  

  def __init__(self, nIn, nOut, stride):
    super(DownsampleA, self).__init__() 
    assert stride == 2    
    self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)   

  def forward(self, x):   
    x = self.avg(x)  
    return torch.cat((x, x.mul(0)), 1)  


class ResNetBasicblock(nn.Module):
  expansion = 1
  """
  RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
  """
  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(ResNetBasicblock, self).__init__()

    self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn_a = nn.BatchNorm2d(planes)

    self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn_b = nn.BatchNorm2d(planes)

    self.downsample = downsample

  def forward(self, x):
    residual = x

    basicblock = self.conv_a(x)
    basicblock = self.bn_a(basicblock)
    basicblock = F.relu(basicblock, inplace=True)

    basicblock = self.conv_b(basicblock)
    basicblock = self.bn_b(basicblock)

    if self.downsample is not None:
      residual = self.downsample(x)
    
    return F.relu(residual + basicblock, inplace=True)


class CifarResNet(object):
  """
  ResNet optimized for the Cifar dataset, as specified in
  https://arxiv.org/abs/1512.03385.pdf
  """
  def __init__(self, block, depth, num_classes):
    """ Constructor
    Args:
      depth: number of layers.
      num_classes: number of classes
      base_width: base width
    """
    super(CifarResNet, self).__init__()
    #super().__init__()

    #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
    assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
    layer_blocks = (depth - 2) // 6
    print ('CifarResNet : Depth : {} , Layers for each block : {}'.format(depth, layer_blocks))

    self.num_classes = num_classes


    self.layers = []

    self.conv_1_3x3 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
    self.layers.append(self.conv_1_3x3)
    self.bn_1 = nn.BatchNorm2d(16)
    self.layers.append(self.bn_1)
    self.relu = nn.ReLU()
    self.layers.append(self.relu)

    list_planes = [16,]*layer_blocks + [32,]*layer_blocks + [64,]*layer_blocks
    list_stride = [1, 2, 2]
    self.inplanes = 16
    
    stage = 'stage'
    for i, planes in enumerate(list_planes):
      stride = 1
      downsample = None
      if i % layer_blocks == 0:
        cur_stage = stage + '_' + str(i//layer_blocks+1)
        stride = list_stride[i//layer_blocks] 
        if stride != 1 or self.inplanes != planes * block.expansion:
          downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)
      # add block into self.layers
      self.layers.append(block(self.inplanes, planes, stride, downsample))
      # update self.inplanes
      if i % layer_blocks == 0:
        self.inplanes = planes * block.expansion
      # set attribute of class
      setattr(self, cur_stage+'_'+str(i%layer_blocks+1), self.layers[-1])

    self.avgpool = nn.AvgPool2d(8)
    self.layers.append(self.avgpool)

    self.classifier = nn.Linear(64*block.expansion, num_classes)


class CifarResNetDDG(nn.Module):
  def __init__(self, model, layers, splits_id, num_splits, delay):
    super(CifarResNetDDG, self).__init__()

    self.splits_id = splits_id
    self.num_splits = num_splits
    self.delay = delay
    self.history_inputs = deque(maxlen=delay+1)
    self.history_outputs = deque(maxlen=delay+1)
    
    self.layers = nn.Sequential(*layers)

    if splits_id == self.num_splits - 1:
      self.classifier = model.classifier

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        #m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        init.kaiming_normal(m.weight)

  def forward(self, x):
    # store inputs for backpropagation
    if self.training:
      self.history_inputs.append(x)

    x = self.layers(x)

    if self.splits_id == self.num_splits-1:
      x = x.view(x.size(0), -1)
      x = self.classifier(x)

    # store outputs for backpropagation 
    if self.training:
      self.history_outputs.append(x)
    return x
  
  def backward(self):
    if self.splits_id == self.num_splits - 1:
      pass
    else:
      if self.delay > 0:
        self.delay -= 1
        return

      if self.delay == 0:
        self.delay -= 1
      prev_output = self.history_outputs.popleft()
      prev_output.backward(self.prev_grad)

  def backup(self, grad):
    self.prev_grad = grad

  def get_grad(self):
    prev_input = self.history_inputs.popleft()
    return prev_input.grad.data


def resnet_ddg(depth, num_classes=10, num_splits=2):
  """
  construct resnet network.
  """
  model = CifarResNet(ResNetBasicblock, depth, num_classes)
  len_layers = len(model.layers) 
  split_depth = math.ceil(len_layers / num_splits) 
  nets = []
  for splits_id in range(num_splits):
    left_idx = splits_id * split_depth
    right_idx = (splits_id+1) * split_depth
    if right_idx > len_layers:
      right_idx = len_layers
    net = CifarResNetDDG(model, model.layers[left_idx:right_idx], splits_id, num_splits, num_splits-1-splits_id)
    nets.append(net) 

  return nets


def resnet_ddg_110(depth, num_classes=10, num_splits=2):
  return resnet_ddg(110, num_classes=num_classes, num_splits=num_splits)


