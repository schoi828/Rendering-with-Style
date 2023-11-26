import pickle, random 
import math, time, platform
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image
from skimage import color
import sys
import os
from torchvision import transforms, datasets
from torchvision.transforms import functional as tvF
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset # For custom datasets

class color_aug(torch.nn.Module):
    def __init__(self):
        pass
    def __call__(self,colored):
        r, g, b = colored.convert('HSV').split()

        r_aug,g_aug = np.random.normal(1,0.35,2)

        r = r.point(lambda i: i * r_aug)

        g = g.point(lambda i: i * g_aug)

        #b = b.point(lambda i: i * b_aug)
        return Image.merge('HSV', (r, g, b)).convert('RGB')

class RandomCrop(object):
    def __init__(self, size):
        assert isinstance(size, (tuple, int))
        self.size = size

    def __call__(self,sketch, colored):
        w, h    = colored.width, colored.height
        #scale   = w / self.size[0] if h > w else h / self.size[1]

        #w, h    = int(w / scale), int(h / scale)
        #colored = colored.resize((w, h), Image.BICUBIC)
        #sketch  = sketch.resize((w, h), Image.BICUBIC)

        x       = np.random.randint(0, w - self.size[0]) if w != self.size[0] else 0
        y       = np.random.randint(0, h - self.size[1]) if h != self.size[1] else 0

        colored = colored.crop((x, y, x + self.size[0], y + self.size[1]))
        #sketch  = sketch.crop((x, y, x + self.size[0], y + self.size[1]))
        sketch = sketch.crop((x, y, x + self.size[0], y + self.size[1]))

        return sketch, colored

def pseudo_uniform(id, a, b):
    return (((id * 1.253 + a * 324.2351 + b * 534.342) * 20147.2312369804) + 0.12949) % (b - a) + a

def real_uniform(id, a, b):
    return random.uniform(a, b)


class RenderDataset(Dataset):
    def __init__(self, args,**kwargs):

        self.img_size = 512
        
        self.root = args.root
        mask_feature = args.mask_feature
        #assert mask_feature in [None,'eye','nose','full'], 'mask feature: eye, nose'
        self.m_path = os.path.join(self.root,'mask')
        if mask_feature is not None:
            self.m_path += '_'+mask_feature
        self.i_path = os.path.join(self.root,args.dataset)
        self.files = [i for i in os.listdir(self.i_path) if i.split('.')[-1] in ['jpg','png']]
        assert len(self.files)>0
        self.crop_size =  args.crop_size
        
        resize = []
        toTensor = [transforms.ToTensor()]
        preprocess = [transforms.ToTensor()]
        
        if self.crop_size > 0:
            resize.append(transforms.CenterCrop(crop_size))
            toTensor.append(transforms.CenterCrop(crop_size))
            proprocess.append(transforms.CenterCrop(crop_size))
        
        resize.append(transforms.Resize(1024))
        toTensor.append(transforms.Resize(512))
        preprocess.append(transforms.Resize(512))
        
        self.resize = transforms.Compose(resize)
        self.toTensor = transforms.Compose(toTensor)
        self.preprocess = transforms.Compose(preprocess)
        #self.ToTensor = transforms.ToTensor()
        #self.center   = transforms.CenterCrop(512)
        
    def __getitem__(self,index):#args,img_path):
        
        name = self.files[index]
        i_path = os.path.join(self.i_path,name)
        m_path = os.path.join(self.m_path,name)
        
        with Image.open(i_path).convert('RGB') as img_full:
            img_full = self.preprocess(img_full)
        with Image.open(m_path).convert('L') as img_mask:
            img_mask = np.array(img_mask)
            img_mask = self.toTensor(img_mask)
        
        return img_full, img_mask#, toTensor(inv_mask).unsqueeze(0)
    
    def __len__(self):
        return len(self.files)

    def enhance_brightness(self, input_size):
        
        random_jitter = [transforms.ColorJitter(brightness=[1, 7], contrast=0.2, saturation=0.2)]
        data_augmentation = [transforms.Resize((input_size, input_size), interpolation=Image.LANCZOS),
                            transforms.ToTensor()]
        self.sketch_transform = transforms.Compose(random_jitter + data_augmentation)

class RGB2ColorSpace(object):
    def __init__(self, color_space):
        self.color_space = color_space

    def __call__(self, img):
        if self.color_space == 'rgb':
            return (img * 2 - 1.)

        img = img.permute(1, 2, 0) # to [H, W, 3]
        if self.color_space == 'lab':
            img = color.rgb2lab(img) # [0~100, -128~127, -128~127]
            img[:,:,0] = (img[:,:,0] - 50.0) * (1 / 50.)
            img[:,:,1] = (img[:,:,1] + 0.5) * (1 / 127.5)
            img[:,:,2] = (img[:,:,2] + 0.5) * (1 / 127.5)
        elif self.color_space == 'hsv':
            img = color.rgb2hsv(img) # [0~1, 0~1, 0~1]
            img = (img * 2 - 1)

        # to [3, H, W]
        return torch.from_numpy(img).float().permute(2, 0, 1) # [-1~1, -1~1, -1~1]

class ColorSpace2RGB(object):
    """
    [-1, 1] to [0, 255]
    """
    def __init__(self, color_space):
        self.color_space = color_space

    def __call__(self, img):
        """numpy array [b, [-1~1], [-1~1], [-1~1]] to target space / result rgb[0~255]"""
        img = img.data.numpy()

        if self.color_space == 'rgb':
            img = (img + 1) * 0.5

        img = img.transpose(0, 2, 3, 1)
        if self.color_space == 'lab': # to [0~100, -128~127, -128~127]
            img[:,:,:,0] = (img[:,:,:,0] + 1) * 50
            img[:,:,:,1] = (img[:,:,:,1] * 127.5) - 0.5
            img[:,:,:,2] = (img[:,:,:,2] * 127.5) - 0.5
            img_list = []
            for i in img:
                img_list.append(color.lab2rgb(i))
            img = np.array(img_list)
        elif self.color_space == 'hsv': # to [0~1, 0~1, 0~1]
            img = (img + 1) * 0.5
            img_list = []
            for i in img:
                img_list.append(color.hsv2rgb(i))
            img = np.array(img_list)

        img = (img * 255).astype(np.uint8)
        return img # [0~255] / [b, h, w, 3]


def rot_crop(x):
    """return maximum width ratio of rotated image without letterbox"""
    x = abs(x)
    deg45 = math.pi * 0.25
    deg135 = math.pi * 0.75
    x = x * math.pi / 180
    a = (math.sin(deg135 - x) - math.sin(deg45 - x))/(math.cos(deg135-x)-math.cos(deg45-x))
    return math.sqrt(2) * (math.sin(deg45-x) - a*math.cos(deg45-x)) / (1-a)

class RandomFRC(transforms.RandomResizedCrop):
    """RandomHorizontalFlip + RandomRotation + RandomResizedCrop 2 images"""
    def __call__(self, img1, img2):
        img1 = tvF.resize(img1, self.size, interpolation=Image.LANCZOS)
        img2 = tvF.resize(img2, self.size, interpolation=Image.LANCZOS)
        if random.random() < 0.5:
            img1 = tvF.hflip(img1)
            img2 = tvF.hflip(img2)
        if random.random() < 0.5:
            rot = random.uniform(-10, 10)
            crop_ratio = rot_crop(rot)
            img1 = tvF.rotate(img1, rot, resample=Image.BILINEAR)
            img2 = tvF.rotate(img2, rot, resample=Image.BILINEAR)
            img1 = tvF.center_crop(img1, int(img1.size[0] * crop_ratio))
            img2 = tvF.center_crop(img2, int(img2.size[0] * crop_ratio))

        i, j, h, w = self.get_params(img1, self.scale, self.ratio)

        # return the image with the same transformation
        return (tvF.resized_crop(img1, i, j, h, w, self.size, self.interpolation),
                tvF.resized_crop(img2, i, j, h, w, self.size, self.interpolation))

def get_loader(args):

    dataset = RenderDataset(args)
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,drop_last=False)
    
    return loader


class LinerTestDataset(Dataset):
    def __init__(self, sketch_path, file_id_list, iv_class_list, cv_class_list,
            override_len=None, sketch_transform=None, **kwargs):
        self.sketch_path = sketch_path

        self.file_id_list = file_id_list # copy

        self.iv_class_list = iv_class_list
        self.cv_class_list = cv_class_list

        self.sketch_transform = sketch_transform
        self.data_len = len(file_id_list)

        if override_len > 0 and self.data_len > override_len:
            self.data_len = override_len

    def __getitem__(self, idx):
        file_id = self.file_id_list[idx]

        iv_tag_class = self.iv_class_list[idx]
        cv_tag_class = self.cv_class_list[idx]

        sketch_path = self.sketch_path / f"{file_id}.png"

        sketch_img = Image.open(sketch_path).convert('L')  # to [1, H, W]
        if self.sketch_transform is not None:
            sketch_img = self.sketch_transform(sketch_img)

        return (sketch_img, file_id, iv_tag_class, cv_tag_class)

    def __len__(self):
        return self.data_len
