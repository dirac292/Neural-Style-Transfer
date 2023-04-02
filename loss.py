import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy
from torchvision.utils import save_image

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ContentLoss(nn.Module):
    def __init__(self,content):
        super(ContentLoss,self).__init__()
        self.content = content.detach() # detach the content from the gradient calculation

    def forward(self,gen): # Input is the generated image\
        self.loss = F.mse_loss(gen,self.content)
        return gen

def gram_mtx(input):
    b,c,h,w = input.size() # B, C, H W
    features = input.view(b*c,h*w)
    G = torch.mm(features,features.t()) # Gram matrix
    return G.div(b*c*h*w) # normalize gram matrix


class StyleLoss(nn.Module):
    def __init__(self,style):
        super(StyleLoss,self).__init__()
        self.style = gram_mtx(style).detach()
    def forward(self,gen):
        G = gram_mtx(gen)
        self.loss = F.mse_loss(G,self.style)
        return gen
    
class Normalization(nn.Module):
    def __init__(self,mean,std):
        super(Normalization,self).__init__()
        self.mean = torch.tensor(mean).view(-1,1,1) # C * 1 * 1
        self.std = torch.tensor(std).view(-1,1,1)
    def forward(self,img):
    # normalize the image
        return (img - self.mean)/self.std