import os
import math
import torch
import argparse
import numpy as np
import torchvision
import model.E.E_v2 as BE
from collections import OrderedDict
from training_utils import *
from model.stylegan1.net import Generator, Mapping #StyleGANv1

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='the training args')
    parser.add_argument('--epoch', type=int, default=200000)
    parser.add_argument('--lr', type=float, default=0.0015)
    parser.add_argument('--beta_1', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--experiment_dir', default=None) #None
    parser.add_argument('--checkpoint_dir', default='./checkpoint/stylegan_v1/ffhq1024/') #None
    parser.add_argument('--config_dir', default=None)
    parser.add_argument('--img_size',type=int, default=1024)
    parser.add_argument('--img_channels', type=int, default=3)# RGB:3 ,L:1
    parser.add_argument('--z_dim', type=int, default=512)
    parser.add_argument('--mtype', type=int, default=1) # StyleGANv1=1, StyleGANv2=2, PGGAN=3, BigGAN=4
    parser.add_argument('--start_features', type=int, default=16) 
    args = parser.parse_args()

    # if not os.path.exists('./result'): os.mkdir('./result')
    # resultPath = args.experiment_dir
    # if resultPath == None:
    #     resultPath = "./result/StyleGAN1-FFHQ1024-Aligned-Img-LossRate"
    #     if not os.path.exists(resultPath): os.mkdir(resultPath)

    # resultPath1_1 = resultPath+"/imgs"
    # if not os.path.exists(resultPath1_1): os.mkdir(resultPath1_1)

    # resultPath1_2 = resultPath+"/models"
    # if not os.path.exists(resultPath1_2): os.mkdir(resultPath1_2)

    # writer_path = os.path.join(resultPath, './summaries')
    # if not os.path.exists(writer_path): os.mkdir(writer_path)
    # writer = tensorboardX.SummaryWriter(writer_path)

    use_gpu = False
    device = torch.device("cuda" if use_gpu else "cpu")

    # train(tensor_writer=writer, args = args)

    model_path = './checkpoint/stylegan_v1/ffhq1024/'
    Gs = Generator(startf=args.start_features, maxf=512, layer_count=int(math.log(args.img_size,2)-1), latent_size=512, channels=3)
    Gs.load_state_dict(torch.load(model_path+'Gs_dict.pth',map_location=torch.device(device)))

    Gm = Mapping(num_layers=int(math.log(args.img_size,2)-1)*2, mapping_layers=8, latent_size=512, dlatent_size=512, mapping_fmaps=512) #num_layers: 14->256 / 16->512 / 18->1024
    Gm.load_state_dict(torch.load(model_path+'Gm_dict.pth',map_location=torch.device(device)))

    Gm.buffer1 = torch.load(model_path+'./center_tensor.pt',map_location=torch.device(device))
    const_ = Gs.const
    const1 = const_.repeat(args.batch_size,1,1,1).to(device)
    layer_num = int(math.log(args.img_size,2)-1)*2 # 14->256 / 16 -> 512  / 18->1024 
    layer_idx = torch.arange(layer_num)[np.newaxis, :, np.newaxis] # shape:[1,18,1], layer_idx = [0,1,2,3,4,5,6。。。，17]
    ones = torch.ones(layer_idx.shape, dtype=torch.float32) # shape:[1,18,1], ones = [1,1,1,1,1,1,1,1]
    coefs = torch.where(layer_idx < layer_num//2, 0.7 * ones, ones) # 18个变量前8个裁剪比例truncation_psi [0.7,0.7,...,1,1,1]

    #Gs.cuda()
    Gm.eval()

    E = BE.BE(startf=args.start_features, maxf=512, layer_count=int(math.log(args.img_size,2)-1), latent_size=512, channels=3)  
    #E.load_state_dict(torch.load('./checkpoint/E_model_ep115000.pth',map_location=torch.device(device)))
    

    # omit RGB layers EAEv2->MSVv2:
    E_dict = torch.load('./checkpoint/E_model_ep115000.pth',map_location=torch.device(device))
    new_state_dict = OrderedDict()
    for (i1,j1),(i2,j2) in zip (E.state_dict().items(),E_dict.items()):
            new_state_dict[i1] = j2 
    E.load_state_dict(new_state_dict)


    ## EAEv2 -> MSVv3
    # E_dict = torch.load('./checkpoint/E_model_ep115000.pth',map_location=torch.device(device))
    # new_state_dict = OrderedDict()
    # new_state_dict2 = OrderedDict()
    # for (i2,j2) in E_dict.items():
    #         if 'blur' in i2:
    #             #print(i2)
    #             pass
    #         else:
    #             new_state_dict[i2]=j2
    # rgb_w = new_state_dict['FromRGB.0.from_rgb.weight']
    # rgb_b = new_state_dict['FromRGB.0.from_rgb.bias']
    # for i1,j1 in new_state_dict.items():
    #     if 'RGB' in i1:
    #         pass
    #     else:
    #         new_state_dict2[i1] = j1
    # new_state_dict2['FromRGB.from_rgb.weight'] = rgb_w
    # new_state_dict2['FromRGB.from_rgb.bias'] = rgb_b
    # #print(len(E.state_dict().keys()))
    # #print(len(E_dict.keys()))
    # #print(len(new_state_dict.keys()))
    # E.load_state_dict(new_state_dict2)

    epoch = 1
    set_seed(epoch)
    z = torch.randn(args.batch_size, args.z_dim) #[32, 512]

    type = args.mtype
    if type == 1:
        with torch.no_grad(): #这里需要生成图片和变量
            w1 = Gm(z,coefs_m=coefs).to(device) #[batch_size,18,512]
            imgs1 = Gs.forward(w1,int(math.log(args.img_size,2)-2)) # 7->512 / 6->256
    elif type == 2:
        with torch.no_grad():
            #use generator
            result_all = generator(z.to(device), **synthesis_kwargs)
            imgs1 = result_all['image']
            w1 = result_all['wp']

    if type != 4:
        const2,w2 = E(imgs1)
    else:
        const2,w2 = E(imgs1, cond_vector)


    if type == 1:
        imgs2=Gs.forward(w2,int(math.log(args.img_size,2)-2))
    elif type == 2 or 3:
        imgs2=generator.synthesis(w2)['image']
    elif type == 4:
        imgs2, _=G(w2, conditions, truncation)
    else:
        print('model type error')

    n_row = args.batch_size
    test_img = torch.cat((imgs1[:n_row],imgs2[:n_row]))*0.5+0.5
    torchvision.utils.save_image(test_img, './ep%d.jpg'%(epoch),nrow=n_row) # nrow=3