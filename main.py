# Optimization method 
# Using trained VGG Net

# L(P,A,X) = alpha * L_content(p,x) + beta * L_style(a,x) 
# p is the content image
# a is the syle image
# x is the generated image

# Paper Reference: https://arxiv.org/pdf/1508.06576.pdf

import os
import uuid
import argparse
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
from loss import ContentLoss,StyleLoss,Normalization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
imsize = 512 if torch.cuda.is_available() else 256

def image_loader(style_img,content_img,input_img,imsize,gr):
   
        
    tr = transforms.Compose([transforms.Resize((imsize,imsize)),transforms.ToTensor()])
    if gr:
        style_image = tr(style_img).unsqueeze(0)
        content_image = tr(content_img).unsqueeze(0)
    else:
        style_image = Image.open(style_img)
        content_image = Image.open(content_img)
    #Image : Channel * height * width
    # After unsqueeze: Batch size * Channel * height * width
        style_image = tr(style_image).unsqueeze(0)
        content_image = tr(content_image).unsqueeze(0)
    if input_img == 'noise':
        input_image = torch.randn(content_image.data.size(),device=device)
    return style_image.to(device,torch.float),content_image.to(device,torch.float),input_image


#Displaying Images
def display_image(tensor,title=None):
    tr = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0) # Remove the batch dimension
    image = tr(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


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


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(model,style_losses,content_losses,content_img,style_img,input_img,num_steps=300,
                       style_weight = 1000000,content_weight = 1):
    
    print("Building the neural style transfer model..")
    optimizer = get_input_optimizer(input_img)
    print("Running optimizer")
    tr = transforms.ToPILImage()
    intermediate_img = []
    
    image = input_img.detach().cpu().clone()
    image_np = image.squeeze(0)
    image_np = tr(image_np)
    intermediate_img.append(image_np)
   # np.transpose(input_img.detach().cpu().numpy(), (1, 2, 0))
    plt.imshow(image_np)
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
            if run[0] % 100 == 0:
                print(f"run {run}")
                print("Style Loss : {:4f} Content Loss: {:4f}".format(style_score.item(),content_score.item()))
                image = input_img.detach().cpu().clone()
                image_np = image.squeeze(0)
                image_np = tr(image_np)
                intermediate_img.append(image_np)
            return style_score + content_score
        optimizer.step(closure)
    input_img.data.clamp_(0,1)
    
    return input_img,intermediate_img

def NST(content_lyr,content,style_lyr,style,input_img,num_steps,style_weight,content_weight,gr):
    
    if not gr:
    
        style_img,content_img,input_img = image_loader(f"{dir}/style_images/{style}",f"{dir}/content_images/{content}",input_img,imsize,False)
        assert style_img.size() == content_img.size()
    else:
        style_img,content_img,input_img = style,content,input_img
        assert style_img.size() == content_img.size()
    
    # Get the VGG 19 model (Set it to evaluation mode)
    cnn = models.vgg19(pretrained = True).features.to(device).eval()

    # VGG net is normalized with mean and std
    cnn_normalization_mean = torch.tensor([0.485,0.456,0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229,0.224,0.225]).to(device)
    
    model,style_losses,content_losses = get_model(cnn,cnn_normalization_mean,cnn_normalization_std,style_img,
                                                  content_img,content_lyr,style_lyr)
    
    output,inter = run_style_transfer(model,style_losses,content_losses,content_img,style_img,
                                input_img,num_steps=num_steps,style_weight = style_weight,content_weight = content_weight)
    
    if gr:
        tr = transforms.ToPILImage()
        image = output.cpu().clone()
        image = image.squeeze(0) # Remove the batch dimension
        image = tr(image)
        return image
    id = uuid.uuid4()
    saved_img = f"{save_dir}/{os.path.splitext(args.style)[0]}_{os.path.splitext(args.content)[0]}_{id}"
    
    save_image(output,f"{saved_img}.jpg")
    
    fig, axs = plt.subplots(nrows=1, ncols=len(inter), figsize=(20, 5))
    for i, img in enumerate(inter):
        axs[i].imshow(img)
        axs[i].axis('off')
        
    filename = f"{saved_img}_sequence.jpg"
    plt.savefig(filename)
    

if __name__ == '__main__':
    
    import random
    random.seed(10)
    
    #Params
    style_lyr = ['conv_1','conv_2','conv_3','conv_4','conv_5']
    content_lyr = ['conv_5']
    dir = './images'
    save_dir = './nst_images'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_lyr', type=list, required=False,default=['conv_5'])
    parser.add_argument('--style_lyr', type=list, required=False,default=['conv_1','conv_2','conv_3','conv_4','conv_5'])
    parser.add_argument('--style', type=str, required=True)
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--display',type = bool,required=False,default=False)
    parser.add_argument('--num_steps',type = int,required=False,default = 500)
    parser.add_argument('--style_weight',type=int,required=False,default=1000000)
    parser.add_argument('--content_weight',type=int,required=False,default=1)
    
    args = parser.parse_args()    
    
    if args.display:
        plt.figure()
        display_image(style_img,title="style image")

        plt.figure()
        display_image(content_img,title="content_image")

        plt.figure()
        display_image(input_img,title='Input Image')
        
        NST(args.content_lyr,args.content,args.style_lyr,args.style,args.input,args.num_steps,args.style_weight,args.content_weight,False)
        
        plt.figure()
        display_image(output,title='Output Image')
        plt.ioff()
        plt.show()
    else:
        NST(args.content_lyr,args.content,args.style_lyr,args.style,args.input,args.num_steps,args.style_weight,args.content_weight,False)
        
  
