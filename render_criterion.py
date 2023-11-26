import torch
import numpy as np
import torch.nn as nn
from torchvision.models import vgg16
from BiSeNet import BiSeNet
from torchvision import transforms
import torch.nn.functional as F

def l2norm(x,y):
    loss = 0
    for x_,y_ in zip(x,y):
        loss+=torch.norm(x_-y_)
    return loss

class Energy(nn.Module):
    def __init__(self,args,k,cp='../face-parsing.PyTorch/res/79999_iter.pth'):
        super().__init__()
        self.per = vgg16(pretrained=True).features[:16]
        self.seg = BiSeNet(n_classes=19)
        self.seg.load_state_dict(torch.load(cp))
        self.per.requires_grad_(False)
        self.seg.requires_grad_(False)
        self.seg.eval()
        self.per.eval()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.pool224 = nn.AdaptiveAvgPool2d(224)
        self.pool512 = nn.AdaptiveAvgPool2d(512)
        self.pool64 = transforms.Resize(64)
        self.pool56 = transforms.Resize(56)

        self.l2 = l2norm#nn.MSELoss()
        self.r_weight = args.r_weight
        self.s_weight = args.s_weight
        self.m_weight = args.m_weight
        self.t_weight = args.t_weight
        self.i_weight = args.i_weight
        self.k = k
        self.X_prev = None
        self.global_mean = args.global_mean
        self.x_m = None
    def forward(self,X,X_full,proj_inp,gt,p,mask):
        
        render_loss = self.render_loss(gt, p, mask)
        consistency_loss = self.consistency_loss(X,X_full,proj_inp,p,(1-mask)) if self.k > 1 else 0
        #self.X_prev = X.detach()
        
        return render_loss,consistency_loss
        
    def consistency_loss(self,X,X_full,proj_inp,projection,inv_mask):
        m_loss = self.mean_loss(X,X_full)*self.m_weight
        t_loss = self.temp_loss(X_full)*self.t_weight
        i_loss = self.inpaint_loss(proj_inp,projection,inv_mask)*self.i_weight if proj_inp is not None else 0
        return m_loss+t_loss+i_loss
    
    def temp_loss(self,x):
        x_t = x[:-1,:].detach()
        prev = torch.zeros(1,x.shape[1],device=x.device)
        x_t = torch.cat([prev,x_t],0)
        return self.l2(x,x_t)    
    """
    def temp_loss(self,x):
        x_t = x[:-1,:].detach()
        prev = torch.zeros(1,x.shape[1],device=x.device) if self.X_prev is None else self.X_prev[-1,:].unsqueeze(0) 
        x_t = torch.cat([prev,x_t],0)
        return self.l2(x,x_t)
    """
    def mean_loss(self,X,X_full):### X_full - X_m
        if not self.global_mean:
            X_m = self.x_m.detach().mean(0).repeat(X.size(0),1) #if self.X_prev is None else torch.cat([self.X_prev,X.detach()],0) 
            return self.l2(X,X_m)
        else:
            return self.l2(X_full,X_full.detach().mean(0).repeat(X.size(0),1))

    def inpaint_loss(self,proj_inp,p,inv_mask):
        
        proj_inp = self.pool512(proj_inp)
        p = self.pool512(p)
        proj_inp = self.pool224(proj_inp)
        p = self.pool224(p)
        inv_mask = self.pool56(inv_mask[:,0].unsqueeze(1))

        proj_features=self.per(proj_inp)*inv_mask
        p_features=self.per(p)*inv_mask
        return self.l2(proj_features,p_features)
        
    def render_loss(self, gt, p, mask):
        
        #gt = gt.mul(0.5).add(0.5)
        p = p.mul(0.5).add(0.5)
        gt = self.normalize(gt)
        p = self.normalize(p)
        
        #gt_512 = self.pool512(gt_render)
        p = self.pool512(p)
        #return self.l2(p,gt)
        #p = p*mask
        #gt = gt*mask
        mask = self.pool64(mask[:,0].unsqueeze(1))
        #mask = mask[:,0].unsqueeze(1)
        gt_features = self.seg(gt,True)*mask
        p_features = self.seg(p,True)*mask
        #print(gt_features.shape)
        s_loss = self.l2(gt_features, p_features)*self.s_weight
        
        gt = self.pool224(gt)
        p = self.pool224(p)
        mask = self.pool56(mask)
        gt_features=self.per(gt)*mask
        p_features=self.per(p)*mask
        #print(gt_features.shape)
        r_loss = self.l2(gt_features, p_features)*self.r_weight
        
        return r_loss+s_loss

class consistency_loss(nn.Module):
    def __init__(self, m_weight,t_weight,i_weight):
        super().__init__()
        self.m_weight = m_weight
        self.t_weight = t_weight
        self.i_weight = i_weight
        self.l2 = nn.MSELoss()
    
    def forward(self,X,X_inp,projection,inv_mask):
        
        m_loss = self.mean_loss(X)*self.m_weight if self.m_weight>0 else 0
        t_loss = self.temp_loss(X)*self.t_weight if self.t_weight>0 else 0
        i_loss = self.inpaint_loss(X_inp,projection)*self.i_weight if self.i_weight>0 else 0
        
        return m_loss+t_loss+i_loss
    
    def temp_loss(self,basis,x):
        x_t = x[:-1]
        x_t = torch.cat([torch.zeros(1,x.shape[1],x.shape[2]),x_t],0)
        return self.l2(x,x_t)
    
    def mean_loss(self,x):
        # x:
        x_m = x.mean(0).repeat(x.size(0),1,1)
        return self.l2(x,x_m)
    
    def inpaint_loss(self,X_inp,projection):
        
        
        
        return None