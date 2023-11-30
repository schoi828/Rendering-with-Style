## Unofficial PyTorch implementation of [Rendering with Style](https://studios.disneyresearch.com/2021/11/30/rendering-with-style-combining-traditional-and-neural-approaches-for-high-quality-face-rendering/)

Since [Rendering with Style](https://studios.disneyresearch.com/2021/11/30/rendering-with-style-combining-traditional-and-neural-approaches-for-high-quality-face-rendering/) utilizes StyleGAN2's latent space, this implementation incorporates codes from [the official Pytorch implementation of StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch).
This work was completed in early 2022, and the README will be updated and polished soon.

<p align="center">
  <img src="rws.png" />
  <img src="RWS.gif" width="427" height="240">
</p>

## Setup
Follow the instruction from [the official Pytorch implementation of StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) to setup StyleGAN2 and download pretrained model ffhq.pkl.

## Train

```
python train_render.py commandline_args.txt
```



