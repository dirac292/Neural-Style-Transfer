# Optimization method 
# Using trained VGG Net

# L(P,A,X) = alpha * L_content(p,x) + beta * L_style(a,x) 
# p is the content image
# a is the syle image
# x is the generated image

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

# device = torch.device("mps" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
imsize = 512 if torch.backends.mps.is_available() else 256



def image_loader(img,imsize):
    tr = transforms.Compose([transforms.Resize((imsize,imsize)),transforms.ToTensor()])
    image = Image.open(img)
    #Image : Channel * height * width
    # After unsqueeze: Batch size * Channel * height * width
    image = tr(image).unsqueeze(0)
    return image.to(device,torch.float)

# print(image_loader('images/content_images/golden_gate.jpg').size())

# Load images
dir = "./images/"


plt.ion()

def imshow(tensor,title=None):
    tr = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0) # Remove the batch dimension
    image = tr(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


# Content Loss
class ContentLoss(nn.Module):
    def __init__(self,content):
        super(ContentLoss,self).__init__()
        self.content = content.detach() # detach the content from the gradient calculation

    def forward(self,gen): # Input is the generated image
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

# Get the VGG 19 model (Set it to evaluation mode)
cnn = models.vgg19(pretrained = True).features.to(device).eval()

# VGG net is normalized with mean and std
cnn_normalization_mean = torch.tensor([0.485,0.456,0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229,0.224,0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self,mean,std):
        super(Normalization,self).__init__()
        self.mean = torch.tensor(mean).view(-1,1,1) # C * 1 * 1
        self.std = torch.tensor(std).view(-1,1,1)
    def forward(self,img):
        # normalize the image
        return (img - self.mean)/self.std


# Hyperparameters
content_lyr = ['conv_4']
style_lyr = ['conv_1','conv_2','conv_3','conv_4','conv_5']

def get_model(cnn,normalization_mean,normalization_std,style_img,content_img,content_lyr,style_lyr):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean,normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
        if isinstance(layer,nn.Conv2d):
            i+=1
            name = f'conv_{i}'
        elif isinstance(layer,nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace = False)
        elif isinstance(layer,nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer,nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer {layer.__class__.__name__}')

        model.add_module(name,layer)

        if name in content_lyr:
            # Add content loss 
            content = model(content_img).detach()
            content_loss = ContentLoss(content)
            model.add_module(f"content_loss{i}",content_loss)
            content_losses.append(content_loss)
        
        if name in style_lyr:
            # Add style loss
            style = model(style_img).detach()
            style_loss = StyleLoss(style)
            model.add_module(f"style_loss{i}",style_loss)
            style_losses.append(style_loss)
        
    for i in range(len(model) - 1, -1 ,-1):
        if((isinstance(model[i],ContentLoss)) or (isinstance(model[i],StyleLoss))):
            break
    
    model = model[:(i + 1)]

    return model,style_losses,content_losses

# print(get_model(cnn,cnn_normalization_mean,cnn_normalization_std,style_img,content_img,content_lyr,style_lyr))





def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(cnn,normalization_mean,normalization_std,content_img,style_img,input_img,num_steps=300,style_weight = 1000000,content_weight = 1):
    print("Building the neural style transfer model..")
    model,style_losses,content_losses = get_model(cnn,normalization_mean,normalization_std,style_img,content_img,content_lyr,style_lyr)
    optimizer = get_input_optimizer(input_img)

    print("Running optimizer")
    run = [0]
    while run[0] <= num_steps:
        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0,1)
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0
            for style_layer in style_losses:
                style_score += (1/5) * style_layer.loss # 1/5 in the paper

            for content_layer in content_losses:
                content_score += content_layer.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1 
            if run[0] % 50 == 0:
                print(f"run {run}")
                print("Style Loss : {:4f} Content Loss: {:4f}".format(style_score.item(),content_score.item()))
                print()
            
            return style_score + content_score
        optimizer.step(closure)
    input_img.data.clamp_(0,1)
    return input_img


style_img = image_loader(dir + "style_images/ben_giles.jpg",imsize)
content_img = image_loader(dir + "content_images/green_bridge.jpeg",imsize)
input_img = torch.randn(content_img.data.size(),device=device)

assert style_img.size() == content_img.size()

plt.figure()
imshow(style_img,title="style image")

plt.figure()
imshow(content_img,title="content_image")

plt.figure()
imshow(input_img,title='Input Image')

output = run_style_transfer(cnn,cnn_normalization_mean,cnn_normalization_std,content_img,style_img,input_img,num_steps=300,style_weight = 100000,content_weight = 10)
save_image(output,f'{"ben" + "bridge"}.jpg')

plt.figure()
imshow(output,title='Output Image')
plt.ioff()
plt.show()


