import os
import math
import torch
import torchvision
import model.E.E_Previous as BE
from model.utils.custom_adam import LREQAdam
import metric.pytorch_ssim as pytorch_ssim
import lpips
import numpy as np
import tensorboardX
import argparse
from model.stylegan1.net import Generator, Mapping #StyleGANv1
import model.stylegan2_generator as model_v2 #StyleGANv2
import model.pggan.pggan_generator as model_pggan #PGGAN
from model.biggan_generator import BigGAN #BigGAN
from training_utils import *
from collections import OrderedDict

def train(tensor_writer = None, args = None):
    type = args.mtype

    model_path = args.checkpoint_dir_GAN
    config_path = args.config_dir

    Gs = Generator(startf=args.start_features, maxf=512, layer_count=int(math.log(args.img_size,2)-1), latent_size=512, channels=3)
    Gs.load_state_dict(torch.load(model_path+'Gs_dict.pth'))

    Gm = Mapping(num_layers=int(math.log(args.img_size,2)-1)*2, mapping_layers=8, latent_size=512, dlatent_size=512, mapping_fmaps=512) #num_layers: 14->256 / 16->512 / 18->1024
    Gm.load_state_dict(torch.load(model_path+'Gm_dict.pth'))

    Gm.buffer1 = torch.load(model_path+'./center_tensor.pt')
    const_ = Gs.const
    const1_ = const_.repeat(args.batch_size,1,1,1).to(device)
    const1 = const1_.detach().clone()
    const1.requires_grad = False
    layer_num = int(math.log(args.img_size,2)-1)*2 # 14->256 / 16 -> 512  / 18->1024 
    layer_idx = torch.arange(layer_num)[np.newaxis, :, np.newaxis] # shape:[1,18,1], layer_idx = [0,1,2,3,4,5,6。。。，17]
    ones = torch.ones(layer_idx.shape, dtype=torch.float32) # shape:[1,18,1], ones = [1,1,1,1,1,1,1,1]
    coefs = torch.where(layer_idx < layer_num//2, 0.7 * ones, ones) # 18个变量前8个裁剪比例truncation_psi [0.7,0.7,...,1,1,1]

    Gs.to(device)
    Gm.eval()

    E = BE.BE(startf=args.start_features, maxf=512, layer_count=int(math.log(args.img_size,2)-1), latent_size=512, channels=3)

    # omit RGB layers EAEv2->MSVv2:
    if args.checkpoint_dir_E != None:
        E_dict = torch.load(args.checkpoint_dir_E,map_location=torch.device(device))
        new_state_dict = OrderedDict()
        for (i1,j1),(i2,j2) in zip (E.state_dict().items(),E_dict.items()):
                new_state_dict[i1] = j2 
        E.load_state_dict(new_state_dict)

    E.to(device)
    writer = tensor_writer
    loss_lpips = lpips.LPIPS(net='vgg').to(device)
    batch_size = args.batch_size
    it_d = 0

    img_list = os.listdir(args.img_dir)
    img_tensor_list = [imgPath2loader(args.img_dir+i,size=args.img_size) for i in img_list]
    imgs1 = torch.stack(img_tensor_list, dim = 0)[:args.batch_size].to(device)
    imgs1 = imgs1*2-1

    #optimize E
    E_optimizer = LREQAdam([{'params': E.parameters()},], lr=args.lr, betas=(args.beta_1, 0.99), weight_decay=0) 

    for epoch in range(0,args.epoch):

        const2, w1 = E(imgs1)
        imgs2 = Gs.forward(w1,int(math.log(args.img_size,2)-2)) # 7->512 / 6->256
        const3, w2 = E(imgs2)

        loss_imgs, loss_imgs_info = space_loss(imgs1,imgs2,lpips_model=loss_lpips)

        #loss AT1
        imgs_medium_1 = imgs1[:,:,:,imgs1.shape[3]//8:-imgs1.shape[3]//8]
        imgs_medium_2 = imgs2[:,:,:,imgs2.shape[3]//8:-imgs2.shape[3]//8]
        loss_medium, loss_medium_info = space_loss(imgs_medium_1,imgs_medium_2,lpips_model=loss_lpips)

        #loss AT2
        imgs_small_1 = imgs1[:,:,\
        imgs1.shape[2]//8+imgs1.shape[2]//32:-imgs1.shape[2]//8-imgs1.shape[2]//32,\
        imgs1.shape[3]//8+imgs1.shape[3]//32:-imgs1.shape[3]//8-imgs1.shape[3]//32]

        imgs_small_2 = imgs2[:,:,\
        imgs2.shape[2]//8+imgs2.shape[2]//32:-imgs2.shape[2]//8-imgs2.shape[2]//32,\
        imgs2.shape[3]//8+imgs2.shape[3]//32:-imgs2.shape[3]//8-imgs2.shape[3]//32]

        loss_small, loss_small_info = space_loss(imgs_small_1,imgs_small_2,lpips_model=loss_lpips)

        E_optimizer.zero_grad()
        loss_msiv = loss_imgs + (loss_medium + loss_small)*0.125
        loss_msiv.backward(retain_graph=True)
        E_optimizer.step()

        #Latent-Vectors
        ## w
        loss_w, loss_w_info = space_loss(w1,w2,image_space = False)

        ## c1
        loss_c1, loss_c1_info = space_loss(const2,const3,image_space = False)

        ## c2
        loss_c2, loss_c2_info = space_loss(const1,const2,image_space = False)

        E_optimizer.zero_grad()
        loss_msLv = (loss_w + loss_c1)*0.0125
        loss_msLv.backward() # retain_graph=True
        E_optimizer.step()

        print('i_'+str(epoch))
        print('[loss_imgs_mse[img,img_mean,img_std], loss_imgs_kl, loss_imgs_cosine, loss_imgs_ssim, loss_imgs_lpips]')
        print('---------ImageSpace--------')
        print('loss_small_info: %s'%loss_small_info)
        print('loss_medium_info: %s'%loss_medium_info)
        print('loss_imgs_info: %s'%loss_imgs_info)
        print('---------LatentSpace--------')
        print('loss_w_info: %s'%loss_w_info)
        print('loss_c1_info: %s'%loss_c1_info)
        print('loss_c2_info: %s'%loss_c2_info)

        it_d += 1
        writer.add_scalar('loss_small_mse', loss_small_info[0][0], global_step=it_d)
        writer.add_scalar('loss_samll_mse_mean', loss_small_info[0][1], global_step=it_d)
        writer.add_scalar('loss_samll_mse_std', loss_small_info[0][2], global_step=it_d)
        writer.add_scalar('loss_samll_kl', loss_small_info[1], global_step=it_d)
        writer.add_scalar('loss_samll_cosine', loss_small_info[2], global_step=it_d)
        writer.add_scalar('loss_samll_ssim', loss_small_info[3], global_step=it_d)
        writer.add_scalar('loss_samll_lpips', loss_small_info[4], global_step=it_d)

        writer.add_scalar('loss_medium_mse', loss_medium_info[0][0], global_step=it_d)
        writer.add_scalar('loss_medium_mse_mean', loss_medium_info[0][1], global_step=it_d)
        writer.add_scalar('loss_medium_mse_std', loss_medium_info[0][2], global_step=it_d)
        writer.add_scalar('loss_medium_kl', loss_medium_info[1], global_step=it_d)
        writer.add_scalar('loss_medium_cosine', loss_medium_info[2], global_step=it_d)
        writer.add_scalar('loss_medium_ssim', loss_medium_info[3], global_step=it_d)
        writer.add_scalar('loss_medium_lpips', loss_medium_info[4], global_step=it_d)

        writer.add_scalar('loss_imgs_mse', loss_imgs_info[0][0], global_step=it_d)
        writer.add_scalar('loss_imgs_mse_mean', loss_imgs_info[0][1], global_step=it_d)
        writer.add_scalar('loss_imgs_mse_std', loss_imgs_info[0][2], global_step=it_d)
        writer.add_scalar('loss_imgs_kl', loss_imgs_info[1], global_step=it_d)
        writer.add_scalar('loss_imgs_cosine', loss_imgs_info[2], global_step=it_d)
        writer.add_scalar('loss_imgs_ssim', loss_imgs_info[3], global_step=it_d)
        writer.add_scalar('loss_imgs_lpips', loss_imgs_info[4], global_step=it_d)

        writer.add_scalar('loss_w_mse', loss_w_info[0][0], global_step=it_d)
        writer.add_scalar('loss_w_mse_mean', loss_w_info[0][1], global_step=it_d)
        writer.add_scalar('loss_w_mse_std', loss_w_info[0][2], global_step=it_d)
        writer.add_scalar('loss_w_kl', loss_w_info[1], global_step=it_d)
        writer.add_scalar('loss_w_cosine', loss_w_info[2], global_step=it_d)
        writer.add_scalar('loss_w_ssim', loss_w_info[3], global_step=it_d)
        writer.add_scalar('loss_w_lpips', loss_w_info[4], global_step=it_d)

        writer.add_scalar('loss_c1_mse', loss_c1_info[0][0], global_step=it_d)
        writer.add_scalar('loss_c1_mse_mean', loss_c1_info[0][1], global_step=it_d)
        writer.add_scalar('loss_c1_mse_std', loss_c1_info[0][2], global_step=it_d)
        writer.add_scalar('loss_c1_kl', loss_c1_info[1], global_step=it_d)
        writer.add_scalar('loss_c1_cosine', loss_c1_info[2], global_step=it_d)
        writer.add_scalar('loss_c1_ssim', loss_c1_info[3], global_step=it_d)
        writer.add_scalar('loss_c1_lpips', loss_c1_info[4], global_step=it_d)

        writer.add_scalar('loss_c2_mse', loss_c2_info[0][0], global_step=it_d)
        writer.add_scalar('loss_c2_mse_mean', loss_c2_info[0][1], global_step=it_d)
        writer.add_scalar('loss_c2_mse_std', loss_c2_info[0][2], global_step=it_d)
        writer.add_scalar('loss_c2_kl', loss_c2_info[1], global_step=it_d)
        writer.add_scalar('loss_c2_cosine', loss_c2_info[2], global_step=it_d)
        writer.add_scalar('loss_c2_ssim', loss_c2_info[3], global_step=it_d)
        writer.add_scalar('loss_c2_lpips', loss_c2_info[4], global_step=it_d)

        writer.add_scalars('Image_Space_MSE', {'loss_small_mse':loss_small_info[0][0],'loss_medium_mse':loss_medium_info[0][0],'loss_img_mse':loss_imgs_info[0][0]}, global_step=it_d)
        writer.add_scalars('Image_Space_KL', {'loss_small_kl':loss_small_info[1],'loss_medium_kl':loss_medium_info[1],'loss_imgs_kl':loss_imgs_info[1]}, global_step=it_d)
        writer.add_scalars('Image_Space_Cosine', {'loss_samll_cosine':loss_small_info[2],'loss_medium_cosine':loss_medium_info[2],'loss_imgs_cosine':loss_imgs_info[2]}, global_step=it_d)
        writer.add_scalars('Image_Space_SSIM', {'loss_small_ssim':loss_small_info[3],'loss_medium_ssim':loss_medium_info[3],'loss_img_ssim':loss_imgs_info[3]}, global_step=it_d)
        writer.add_scalars('Image_Space_Lpips', {'loss_small_lpips':loss_small_info[4],'loss_medium_lpips':loss_medium_info[4],'loss_img_lpips':loss_imgs_info[4]}, global_step=it_d)
        writer.add_scalars('Latent Space W', {'loss_w_mse':loss_w_info[0][0],'loss_w_mse_mean':loss_w_info[0][1],'loss_w_mse_std':loss_w_info[0][2],'loss_w_kl':loss_w_info[1],'loss_w_cosine':loss_w_info[2]}, global_step=it_d)
        writer.add_scalars('Latent Space C1', {'loss_c_mse':loss_c1_info[0][0],'loss_c_mse_mean':loss_c1_info[0][1],'loss_c_mse_std':loss_c1_info[0][2],'loss_c_kl':loss_c1_info[1],'loss_c_cosine':loss_c1_info[2]}, global_step=it_d)
        writer.add_scalars('Latent Space C2', {'loss_c_mse':loss_c2_info[0][0],'loss_c_mse_mean':loss_c2_info[0][1],'loss_c_mse_std':loss_c2_info[0][2],'loss_c_kl':loss_c2_info[1],'loss_c_cosine':loss_c2_info[2]}, global_step=it_d)


        if epoch % 100 == 0:
            n_row = batch_size
            test_img = torch.cat((imgs1[:n_row],imgs2[:n_row]))*0.5+0.5
            torchvision.utils.save_image(test_img, resultPath1_1+'/ep%d.jpg'%(epoch),nrow=n_row) # nrow=3
            with open(resultPath+'/Loss.txt', 'a+') as f:
                print('i_'+str(epoch),file=f)
                print('[loss_imgs_mse[img,img_mean,img_std], loss_imgs_kl, loss_imgs_cosine, loss_imgs_ssim, loss_imgs_lpips]',file=f)
                print('---------ImageSpace--------',file=f)
                print('loss_small_info: %s'%loss_small_info,file=f)
                print('loss_medium_info: %s'%loss_medium_info,file=f)
                print('loss_imgs_info: %s'%loss_imgs_info,file=f)
                print('---------LatentSpace--------',file=f)
                print('loss_w_info: %s'%loss_w_info,file=f)
                print('loss_c1_info: %s'%loss_c1_info,file=f)
                print('loss_c2_info: %s'%loss_c2_info,file=f)
            for i,j in enumerate(w1):
                torch.save(j.unsqueeze(0),resultPath1_2+'/i%d-w%d.pt'%(i,epoch))
            #torch.save(Gm.buffer1,resultPath1_2+'/center_tensor_ep%d.pt'%epoch)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='the training args')
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.0015)
    parser.add_argument('--beta_1', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--experiment_dir', default='./result/StyleGAN1-FFHQ1024-Aligned-realImgEmbedding-E2W-ep115000-2') #None
    parser.add_argument('--checkpoint_dir_GAN', default='./checkpoint/stylegan_v1/ffhq1024/') #None  ./checkpoint/stylegan_v1/ffhq1024/ or ./checkpoint/stylegan_v2/stylegan2_ffhq1024.pth
    parser.add_argument('--config_dir', default=None) # BigGAN needs it
    parser.add_argument('--checkpoint_dir_E', default='./checkpoint/E/styleGANv1_EAE_ep115000.pth')
    parser.add_argument('--img_dir', default='./checkpoint/real_img/')
    parser.add_argument('--img_size',type=int, default=1024)
    parser.add_argument('--img_channels', type=int, default=3)# RGB:3 ,L:1
    parser.add_argument('--z_dim', type=int, default=512)
    parser.add_argument('--mtype', type=int, default=1) # StyleGANv1=1, StyleGANv2=2, PGGAN=3, BigGAN=4
    parser.add_argument('--start_features', type=int, default=16)  # 16->1024 32->512 64->256
    args = parser.parse_args()

    if not os.path.exists('./result'): os.mkdir('./result')
    resultPath = args.experiment_dir
    if resultPath == None:
        resultPath = "./result/XXX"
    if not os.path.exists(resultPath): os.mkdir(resultPath)

    resultPath1_1 = resultPath+"/imgs"
    if not os.path.exists(resultPath1_1): os.mkdir(resultPath1_1)

    resultPath1_2 = resultPath+"/models"
    if not os.path.exists(resultPath1_2): os.mkdir(resultPath1_2)

    writer_path = os.path.join(resultPath, './summaries')
    if not os.path.exists(writer_path): os.mkdir(writer_path)
    writer = tensorboardX.SummaryWriter(writer_path) 

    use_gpu = True
    device = torch.device("cuda" if use_gpu else "cpu")

    train(tensor_writer=writer, args = args)
