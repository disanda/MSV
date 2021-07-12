

# MTV: Multi-Type Vectors for Image Embedding with Latent Representation

![Python 3.6.9](https://img.shields.io/badge/python-3.6.9-blue.svg?style=plastic)
![PyTorch 1.5.1](https://img.shields.io/badge/pytorch-1.5.1-blue.svg?style=plastic) 
![Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg?style=plastic)

<center class="half">
<img src="./images/cxx1.gif" width = "128" height = "128" alt="cxx1" align=center />
<img src="./images/cxx2.gif" width = "128" height = "128" alt="cxx1" align=center />
<img src="./images/msk.gif" width = "128" height = "128" alt="cxx1" align=center />
<img src="./images/dy.gif" width = "128" height = "128" alt="cxx1" align=center />
<img src="./images/zy.gif" width = "128" height = "128" alt="cxx1" align=center />
</center>




>This is the official code release for MtV: Multi-Type Vectors for Image Embedding with Latent Representation. The code contains a set of encoders for matching  pre-trained GANs (PGGAN, StyleGANv1, StyleGANv2, BigGAN)  via multi-scale vectors.

## Setup

###   Encoders

​	

###   Pre-Trained GANs

- StyleGAN_V1:
  - Cat 256:
    - ./checkpoint/stylegan_V1/cat/cat256_Gs_dict.pth
    - ./checkpoint/stylegan_V1/cat/cat256_Gm_dict.pth
    - ./checkpoint/stylegan_V1/cat/cat256_tensor.pt
  - Car 256:
  - Bedroom 256:
- StyleGAN_V2:
  - FFHQ 1024:
    - ./checkpoint/stylegan_V2/stylegan2_ffhq1024.pth
- PGGAN:
  - Horse 256:
    - ./checkpoint/PGGAN/
- BigGAN:
  - Image-Net 256:
    - ./checkpoint/biggan/256/G-256.pt
    - ./checkpoint/biggan/256/biggan-deep-256-config.json


## Usage

###  Options

```python
    parser = argparse.ArgumentParser(description='the training args')
    parser.add_argument('--epoch', type=int, default=200000)
    parser.add_argument('--lr', type=float, default=0.0015)
    parser.add_argument('--beta_1', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--experiment_dir', default='./result/StyleGAN2-face1024-modelv3-Aligned-INnoAffine-Gall2cuda') #None
    parser.add_argument('--checkpoint_dir', default='./checkpoint/stylegan_v2/stylegan2_ffhq1024.pth') #None
    parser.add_argument('--img_size',type=int, default=1024)
    parser.add_argument('--img_channels', type=int, default=3)# RGB:3 ,L:1
    parser.add_argument('--z_dim', type=int, default=512)
    parser.add_argument('--mtype', type=int, default=2) # StyleGANv1=1, StyleGANv2=2, PGGAN=3, BigGAN=4
```

###  Training Encoders

```python
>python E_A.py
```



For misaligned-image by Grad-CAM

```python
>python E_MA.py
```




##  Acknowledgements

Pre-trained GANs:

> StyleGANv1: https://github.com/podgorskiy/StyleGan.git, 
> ( Converting  code for official pre-trained model  is here: https://github.com/podgorskiy/StyleGAN_Blobless.git)
> StyleGANv2 and PGGAN: https://github.com/genforce/genforce.git
> BigGAN: https://github.com/huggingface/pytorch-pretrained-BigGAN

Comparing Works:

> In-Domain GAN: https://github.com/genforce/idinvert_pytorch
> pSp: https://github.com/eladrich/pixel2style2pixel
> ALAE: https://github.com/podgorskiy/ALAE.git

Ratelted Works:

> Grad-CAM & Grad-CAM++: https://github.com/yizt/Grad-CAM.pytorch
> SSIM Index: https://github.com/Po-Hsun-Su/pytorch-ssim

We express our thanks to above authors.

## License

The code of this repository is released under the [Apache 2.0](LICENSE) license.<br>
The directory `netdissect` is a derivative of the [GAN Dissection][gandissect] project, and is provided under the MIT license.<br>
The directories `models/biggan` and `models/stylegan2` are provided under the MIT license.


## BibTeX