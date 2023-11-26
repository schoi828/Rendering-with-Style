import os
import re
from typing import List, Optional
import argparse
import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import legacy
import torch.nn as nn
import pickle
import operator
from training.networks import *
import torch.optim as optim
#from render_loader import *
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image
import cv2
import math
import json
import random
from render_loader import get_loader

def get_images(args,img_path):
    
    target_mask = ['skin','hat','nose','hair']#,'nose','right_brow','left_brow','top_lip','bottom_lip','beard','glasses','facewear']
    img_size = 512 #320
    resize = transforms.Compose([
                            transforms.CenterCrop(img_size),
                            transforms.Resize(1024)
                            ]) ##centercrop???
    
    toTensor = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.CenterCrop(img_size),
                            transforms.Resize(512)
                            ]) ##centercrop???
    preprocess = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.CenterCrop(img_size),
                            transforms.Resize(512),
                            #transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
                            ])
    id = '0'*(6-len(str(args.id)))+str(args.id)
    image = id+'.png'
    seg = id+'_seg.png'
    i_path = '../han_ori/neutral/facial_expression.0164.jpg'#turntable_v02.0006.jpg'
    #'../han/aligned2/1_facial_expression.0001_0.jpg'#sample.png'#args.dataset+'/'+image
    seg_path = '../han_ori/mask_full/facial_expression.0164.jpg'#turntable_v02.0006.jpg'#lighting.0189.jpg'#
    #'../han/mask_eye/1_facial_expression.0001_0.jpg'#'sample_seg.png'#args.dataset+'/'+seg
    with Image.open(i_path).convert('RGB').resize((512,512)) as full:
        ori = resize(full)
        ori.save(os.path.join(img_path,str(args.id)+'_ori_'+'.png'))
        img_full = preprocess(full)
    with Image.open(seg_path).convert('RGB') as mask:
        img_masked = np.array(mask)*255

    #img_masked = (img_mask == target_mask[0])
    
    #for cl in target_mask[0:]:
    #    img_masked += (img_mask == cl)
        #print(np.max(img_masked))
    #img_seg = img_full*img_masked
    inv_mask = ~(img_masked)
    img_masked = toTensor(img_masked)
    inv_mask = toTensor(inv_mask)
    img_ori_masked = img_full*(img_masked)
    img_ori_masked = transforms.Resize(1024)(img_ori_masked)
    save_image(img_ori_masked,os.path.join(img_path,str(args.id)+'_masked'+'.png'))
    img_inv_masked = img_full*(inv_mask)
    img_inv_masked = transforms.Resize(1024)(img_inv_masked)
    save_image(img_inv_masked,os.path.join(img_path,str(args.id)+'_invmasked'+'.png'))

    #inv_masked = img_full*inv_mask
    return img_full.unsqueeze(0), img_masked.unsqueeze(0), inv_mask.unsqueeze(0)
    
def train(args):
    img_path = os.path.join(args.save_path,args.name,'img')
    ckp_path = os.path.join(args.save_path,args.name,'checkpoints')
    os.makedirs(img_path,exist_ok=True)
    os.makedirs(ckp_path,exist_ok=True)
    
    if args.resume > 0:
        with open(os.path.join(ckp_path,'commandline_args.txt'), 'r') as f:
            args.__dict__ = json.load(f)
    else:
        with open(os.path.join(ckp_path,'commandline_args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    
    device = torch.device('cuda')
    with open('ffhq.pkl', 'rb') as f:
        saved = pickle.load(f)
        G = saved['G_ema'].cuda()  # torch.nn.Module
        G.eval()
        G.requires_grad_(False)
        if args.dis_loss or args.blend_loss:
            D = saved['D'].cuda()
            D.eval()
            D.requires_grad_(False)
        
    m = G.mapping
    g = G.synthesis    

    w_dim = 512
    img_resolution = 1024
    img_channels = 3
    
    s = SynthesisNetwork(
            w_dim,                      # Intermediate latent (W) dimensionality.
            img_resolution,             # Output image resolution.
            img_channels,               # Number of color channels.
        )
    s.load_state_dict(g.state_dict())
    s = s.to(device)
    for i in s.parameters():
        i.requires_grad = False
    
    z = torch.randn(args.n,512).to(device)
    w = m(z,None)[:,1]#truncation_psi=args.trunc
    
    del m
    del G
    
    if args.serial_path is None:
        
        img_full, img_mask, inv_mask = get_images(args,img_path)
        img_full = img_full.to(device)
        img_mask = img_mask.to(device)
        inv_mask = inv_mask.to(device)
        k = 1
    else:
        dataloader = get_loader(args)
        k = len(dataloader.dataset)
    
    res = 4
    i = 1
    w_i = 0
    basis = None
    rgb = False
    
    from render_criterion import Energy

    #r_loss = render_loss(args.r_weight,args.s_weight).to(device)
    
    if k == 1:
        args.t_weight = 0
        args.m_weight = 0
        args.i_weight = 0
        
    energy = Energy(args,k=1).to(device)
    #c_loss = consistency_loss(args.m_weight,args.t_weight,args.s_weight)
    
    s_basis = dict()
    rgb_w = w[:k,:]
    
    layers_n = 26 if args.all_rgb else 18
    
    c = layers_n*args.c 
    if args.c == 64: c -=32 + 32*args.all_rgb 
    
    while res <= 1024:
        name = '.conv'+str(i) if not rgb else '.torgb'
        rgb = True if ((res ==args.rgb_layer or args.all_rgb) and i == 1) else False
        attr = 'b'+str(res)+name+'.affine'
        fc = operator.attrgetter(attr)(s)
        w_o = fc(w)
        if args.trunc < 1: 
            w_m = w_o.mean(0,keepdim=True)
            w_o = w_m+args.trunc*(w_o-w_m)
        print(attr,w_o.shape)
        
        if args.no_segment:
            basis = w_o if basis is None else torch.cat([basis,w_o],1)
        else:
            w_o = w_o.view(1,args.n,math.ceil(w_o.shape[1]/args.c),-1)
            s_basis[w_i] = w_o
            #print(w_o.view(args.n,int(w_o.shape[1]/args.c),-1).shape)
        #print(attr)
        if i > (0 + 1*args.all_rgb) and (args.all_rgb or rgb == False): 
            res*=2
            i=0
        else:
            i+=1
        w_i+=1
    
    if args.no_segment: basis = basis.unsqueeze(0)

    if args.resume > 0:
        weight = torch.load(os.path.join(ckp_path,str(epoch)+'.pth'))
    else:
        #weight = torch.randn(k,args.n*args.c,1,requires_grad=True,device=device)
        #w = torch.randn(1,size)
        #s = torch.split(w, int(size/18),1)
        segment = 6560 if args.no_segment else c
        w_avg = np.random.normal(0,1,[k,args.n,segment])
        #weight = nn.Parameter(weight).to(device)
        weight = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    optimizer = optim.Adam([weight],lr=args.lr)
    
    softmax = nn.Softmax(dim=1)
    
    if args.blend_loss:
        mask_resize = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize(1024)
                                ]) ##centercrop???
        normalize = transforms.Compose([
                                transforms.Resize(1024),
                                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        
        ])
        #with Image.open('sample_eroded.png') as mask_eroded:
        #    mask_eroded = np.array(mask_eroded)*255
        mask_eroded= img_mask    
        #print(np.min(mask_eroded),np.max(mask_eroded),np.mean(mask_eroded))
        mask_eroded = mask_resize(mask_eroded).unsqueeze(0).to(device)
        #print(torch.min(mask_eroded),torch.max(mask_eroded),torch.mean(mask_eroded))
        ori_tensor = normalize(img_full)
    
    noise_mode = 'random' if args.random_noise else 'const'
    
    for epoch in range(max(0,args.resume),args.max_epochs):
        
        #for idx, img_full, img_mask in enumerate(dataloader):
        optimizer.zero_grad()
        weight_soft = softmax(weight)
        proj_inp = None
        if args.no_segment:
            X = weight_soft*basis
            X = X.sum(1)
        
            if k > 1:
                X_inp = weight_soft[:,:,0].detach().unsqueeze(-1)*basis
                X_inp = X_inp.sum(1)
                proj_inp = s(X_inp,s=True,rgb=rgb_w,noise_mode=noise_mode,rgb_res=args.rgb_layer,all_rgb=args.all_rgb)
        else:
            X = None
            start = 0
            for i in s_basis:
                end = (i+1)*args.c# if i < len(s_basis)-1 else -1
                #print(i,s_basis[i].shape,weight_soft[:,:,start:end].unsqueeze(2).shape)
                x = s_basis[i]*weight_soft[:,:,start:end].unsqueeze(2)
                x = x.sum(1).flatten(1,-1)
                #print(i,x.shape)
                #print(i,x.shape)
                start = end
                X = x if X is None else torch.cat([X,x],1)
                
                if k > 1:
                    pass
        
        #print('X',X.shape)
        proj = s(X,s=True,rgb=rgb_w,noise_mode=noise_mode,rgb_res=args.rgb_layer,all_rgb=args.all_rgb)

        loss,_ = energy(X,proj_inp,img_full,proj,img_mask)
        
        if args.dis_loss:
            p_logits = D(proj,None)
            loss_d=torch.nn.functional.softplus(p_logits).mean().mul(args.d_weight)#composite blending loss?
            loss+=loss_d
        else:
            loss_d = torch.zeros(1)
        
        if args.blend_loss:
            blend_mask = transforms.Resize(1024)(img_mask)*1.0
            blend = blend_mask*ori_tensor + (1-blend_mask)*proj
            b_logits = D(blend,None)
            loss_b=torch.nn.functional.softplus(b_logits).mean().mul(args.b_weight)#composite blending loss?
            loss+=loss_b
        else:
            loss_b = torch.zeros(1)
            #save_image(blend.add(1).mul(0.5),'blend_tensor.png')

        #render_loss = r_loss(img_full,proj,img_mask)
        
        #consistency_loss = c_loss(X,X_inp,proj,inv_mask)
        
        #loss = render_loss+consistency_loss
        
        loss.backward()
        
        optimizer.step()
        
        if epoch % args.print_every == 0:
            print(epoch,'running loss: {:.4f}, D_loss : {:.4f}, B_loss : {:.4f}'.format(loss.item(),loss_d.item(),loss_b.item()))
        
        """
        for i, data in enumerate(dataloader):
            
            ks, img, mask = data
            weight_soft = softmax(weight[ks[0]:ks[1]])
            
            X = weight_soft*basis
            X = X.sum(1)
            
            proj = s(X,s=True,rgb=rgb_w)
        """    
         
        if epoch % args.save_every == 0:
            
            img = (proj.detach().permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img_pil = PIL.Image.fromarray(img[0].cpu().numpy(),'RGB')
            img_pil.save(
                os.path.join(img_path,str(args.id)+'_'+str(epoch)+'.png'))
            
            img_array=np.array(img_pil)
            img_array=cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            ori_array=img_full[0].permute(1, 2, 0).cpu().numpy()*255
            ori_array=cv2.cvtColor(ori_array, cv2.COLOR_RGB2BGR)
            ori_array=cv2.resize(ori_array, (1024,1024))
            #print(img_mask.squeeze(0).permute(1,2,0).cpu().numpy()*1)
            
            #mask_array = cv2.resize(img_mask.squeeze(0).permute(1,2,0).cpu().numpy()*1,(1024,1024))
            mask_array = 1.0*img_mask.permute(0, 2, 3, 1)[0].cpu().numpy()
            #1.0*np_mask#transforms.Resize(1024)(img_mask)).squeeze(0).permute(1,2,0).cpu().numpy()
            mask_array = cv2.resize(mask_array,(1024,1024))
            kernel = np.ones((1,1),np.uint8)
            #print(mask_array.shape)
            mask_array = cv2.erode(mask_array,kernel,iterations = 30)
            mask_array = cv2.GaussianBlur(mask_array,(5,5),0)
            #cv2.imwrite(os.path.join(img_path,'sample_eroded.png'),mask_array)
            blend = mask_array*ori_array+(1-mask_array)*img_array
            cv2.imwrite(os.path.join(img_path,str(args.id)+'_'+str(epoch)+'_blended.png'),blend)

            torch.save(weight,os.path.join(ckp_path,str(epoch)+'.pth'))
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--n', type=int, default=64, help="n basis")
    parser.add_argument('--c', type=int, default=32, help="segments per layer")
    parser.add_argument('--trunc', type=float, default=0.8)
    parser.add_argument('--save_path', type=str, default='out')
    parser.add_argument('--save_every', type=int, default=300)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=10000)
    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--dataset', type=str, default='../dataset_100000')
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--r_weight', type=float, default=1)
    parser.add_argument('--s_weight', type=float, default=0.01)
    parser.add_argument('--m_weight', type=float, default=0.0001)
    parser.add_argument('--t_weight', type=float, default=0.0001)
    parser.add_argument('--i_weight', type=float, default=0.1)
    parser.add_argument('--d_weight', type=float, default=2)
    parser.add_argument('--b_weight', type=float, default=2)
    parser.add_argument('--no_segment', action='store_true')
    parser.add_argument('--dis_loss', action='store_true')
    parser.add_argument('--blend_loss', action='store_true')
    parser.add_argument('--random_noise', action='store_true')
    parser.add_argument('--all_rgb', action='store_true')
    parser.add_argument('--rgb_layer',  type=int, default=4, choices=[4, 8, 16, 32, 64, 128, 256, 512, 1024])
    parser.add_argument('--serial_path', type=str, default=None)
    parser.add_argument('--mask_feature', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=-1)



    args = parser.parse_args()

    
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    train(args)