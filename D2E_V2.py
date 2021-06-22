import os
import torch
import torch.nn as nn
import torchvision
import model.E.E_v3 as BE
from model.utils.custom_adam import LREQAdam
import lpips
from metric.grad_cam import GradCAM, GradCamPlusPlus, GuidedBackPropagation, mask2cam
import tensorboardX
import numpy as np
import argparse
from model.stylegan1.net import Generator, Mapping #StyleGANv1
import model.stylegan2_generator as model_v2 #StyleGANv2
import model.pggan.pggan_generator as model_pggan #PGGAN
from model.biggan_generator import BigGAN #BigGAN
from training_utils import *

#torch.backends.cudnn.enabled = True
#torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.deterministic = False # faster

def train(tensor_writer = None, args = None):
    type = args.mtype

    if type == 1: # StyleGAN1

        Gs = Generator(startf=64, maxf=512, layer_count=7, latent_size=512, channels=3)
        Gs.load_state_dict(torch.load('./checkpoint/cat/cat256_Gs_dict.pth'))

        Gm = Mapping(num_layers=14, mapping_layers=8, latent_size=512, dlatent_size=512, mapping_fmaps=512) #num_layers: 14->256 / 16->512 / 18->1024
        Gm.load_state_dict(torch.load('./checkpoint/cat/cat256_Gm_dict.pth'))

        Gm.buffer1 = torch.load('./checkpoint/cat/cat256_tensor.pt')
        const_ = Gs.const
        const1 = const_.repeat(args.batch_size,1,1,1).cuda()
        layer_num = 14 # 14->256 / 16 -> 512  / 18->1024 
        layer_idx = torch.arange(layer_num)[np.newaxis, :, np.newaxis] # shape:[1,18,1], layer_idx = [0,1,2,3,4,5,6。。。，17]
        ones = torch.ones(layer_idx.shape, dtype=torch.float32) # shape:[1,18,1], ones = [1,1,1,1,1,1,1,1]
        coefs = torch.where(layer_idx < layer_num//2, 0.7 * ones, ones) # 18个变量前8个裁剪比例truncation_psi [0.7,0.7,...,1,1,1]

        Gs.cuda()
        Gm.eval()

    elif type == 2:  # StyleGAN2

        generator = model_v2.StyleGAN2Generator(resolution=256).to(device)
        checkpoint = torch.load('./checkpoint/stylegan2_horse256.pth') #map_location='cpu'
        if 'generator_smooth' in checkpoint: #default
            generator.load_state_dict(checkpoint['generator_smooth'])
        else:
            generator.load_state_dict(checkpoint['generator'])
        synthesis_kwargs = dict(trunc_psi=0.7,trunc_layers=8,randomize_noise=False)
        Gs = generator.synthesis
        Gm = generator.mapping
        const_r = torch.randn(args.batch_size)
        const1 = Gs.early_layer(const_r) #[n,512,4,4]


    elif type == 3:  # PGGAN

        generator = model_pggan.PGGANGenerator(resolution=256).to(device)
        checkpoint = torch.load('./checkpoint/pggan_horse256.pth') #map_location='cpu'
        if 'generator_smooth' in checkpoint: #默认是这个
            generator.load_state_dict(checkpoint['generator_smooth'])
        else:
            generator.load_state_dict(checkpoint['generator'])
        const1 = torch.tensor(0)


    elif type == 4:

        cache_path = './checkpoint/biggan/256/G-256.pt'
        resolved_config_file = './checkpoint/biggan/256/biggan-deep-256-config.json'
        config = BigGANConfig.from_json_file(resolved_config_file)
        generator = BigGAN(config)
        generator.load_state_dict(torch.load(cache_path))

    else:
        print('error')
        return


    E = BE.BE(startf=64, maxf=512, layer_count=7, latent_size=512, channels=3)
    #E.load_state_dict(torch.load('/_yucheng/myStyle/myStyle-v1/EAE-car-cat/result/EB_cat_cosine_v2/E_model_ep80000.pth'))
    E.cuda()
    writer = tensor_writer

    E_optimizer = LREQAdam([{'params': E.parameters()},], lr=0.0015, betas=(0.0, 0.99), weight_decay=0) 
    #用这个adam不会报错:RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
    loss_lpips = lpips.LPIPS(net='vgg').to('cuda')

    batch_size = args.batch_size
    it_d = 0

    #vgg16->Grad-CAM
    vgg16 = torchvision.models.vgg16(pretrained=True).cuda()
    final_layer = None
    for name, m in vgg16.named_modules():
        if isinstance(m, nn.Conv2d):
            final_layer = name
    grad_cam_plus_plus = GradCamPlusPlus(vgg16, final_layer)
    gbp = GuidedBackPropagation(vgg16)


    it_d = 0
    for epoch in range(0,250001):
        set_seed(epoch%30000)
        z = torch.randn(batch_size, args.z_dim) #[32, 512]

        if type == 1:
            with torch.no_grad(): #这里需要生成图片和变量
                w1 = Gm(z,coefs_m=coefs).cuda() #[batch_size,18,512]
                imgs1 = Gs.forward(w1,6) # 7->512 / 6->256
        elif type == 2:
            with torch.no_grad():
                result_all = generator(z.cuda(), **synthesis_kwargs)
                imgs1 = result_all['image']
                w1 = result_all['wp']
        elif type == 3:
            with torch.no_grad(): #这里需要生成图片和变量
                w1 = z.cuda()
                result_all = generator(w1)
                imgs1 = result_all['image']
        elif type == 4:
            z = truncated_noise_sample(truncation=0.4, batch_size=batch_size, seed=epoch%30000)
            #label = np.random.randint(1000,size=batch_size) # 生成标签
            flag = np.random.randint(1000)
            label = np.ones(batch_size)
            label = flag * label
            label = one_hot(label)
            w1 = torch.tensor(z, dtype=torch.float).cuda()
            conditions = torch.tensor(label, dtype=torch.float).cuda() # as label
            truncation = torch.tensor(synthesis_kwargs, dtype=torch.float).cuda()
            with torch.no_grad(): #这里需要生成图片和变量
                imgs1, const1 = generator(w1, conditions, truncation) # const1 are conditional vectors in BigGAN

        if type != 4:
            const2,w2 = E(imgs1)
        else:
            const2,w2 = E(imgs1, cond_vector)

        if type == 1:
            imgs2=Gs.forward(w2,6)
        elif type == 2 or 3:
            imgs2=Gs(w2)['image']
        elif type == 4:
            imgs2, _=G(w2, conditions, truncation)
        else:
            print('model type error')
            return
        
        E_optimizer.zero_grad()

#Latent Space
    ##--C
        loss_c, loss_c_info = space_loss(const1,const2,image_space = False)
        E_optimizer.zero_grad()
        loss_c.backward(retain_graph=True)
        E_optimizer.step()

    ##--W
        loss_w, loss_w_info = space_loss(w1,w2,image_space = False)
        E_optimizer.zero_grad()
        loss_w.backward(retain_graph=True)
        E_optimizer.step()

#Image Space
        mask_1 = grad_cam_plus_plus(imgs1.detach().clone(),None) #[c,1,h,w]
        mask_2 = grad_cam_plus_plus(imgs2.detach().clone(),None)
        #imgs1.retain_grad()
        #imgs2.retain_grad()
        imgs1_ = imgs1.detach().clone()
        imgs1_.requires_grad = True
        imgs2_ = imgs2.detach().clone()
        imgs2_.requires_grad = True
        grad_1 = gbp(imgs1_) # [n,c,h,w]
        grad_2 = gbp(imgs2_)
        heatmap_1,cam_1 = mask2cam(mask_1,imgs1)
        heatmap_2,cam_2 = mask2cam(mask_2,imgs2)

    ##--Mask_Cam
        mask_1 = mask_1.cuda().float()
        mask_1.requires_grad=True
        mask_2 = mask_2.cuda().float()
        mask_2.requires_grad=True
        loss_mask, loss_mask_info = space_loss(mask_1,mask_2,lpips_model=loss_lpips)

        E_optimizer.zero_grad()
        loss_mask.backward(retain_graph=True)
        E_optimizer.step()

    ##--Grad
        grad_1 = grad_1.cuda().float()
        grad_1.requires_grad=True
        grad_2 = grad_2.cuda().float()
        grad_2.requires_grad=True
        loss_grad, loss_grad_info = space_loss(grad_1,grad_2,lpips_model=loss_lpips)

        E_optimizer.zero_grad()
        loss_grad.backward(retain_graph=True)
        E_optimizer.step()

    ##--Image
        loss_imgs, loss_imgs_info = space_loss(imgs1.detach().clone(),imgs2.detach().clone(),lpips_model=loss_lpips)
        E_optimizer.zero_grad()
        loss_imgs.backward(retain_graph=True)
        E_optimizer.step()

    ##--Grad_CAM from mask
        cam_1 = cam_1.cuda().float()
        cam_1.requires_grad=True
        cam_2 = cam_2.cuda().float()
        cam_2.requires_grad=True
        loss_Gcam, loss_Gcam_info = space_loss(cam_1,cam_2,lpips_model=loss_lpips)

        E_optimizer.zero_grad()
        loss_Gcam.backward(retain_graph=True)
        E_optimizer.step()

        print('i_'+str(epoch))
        print('[loss_imgs_mse[img,img_mean,img_std], loss_imgs_ssim, loss_imgs_cosine, loss_kl_imgs, loss_imgs_lpips]')
        print('---------ImageSpace--------')
        print('loss_mask_info: %s'%loss_mask_info)
        print('loss_grad_info: %s'%loss_grad_info)
        print('loss_imgs_info: %s'%loss_imgs_info)
        print('loss_Gcam_info: %s'%loss_Gcam_info)
        print('---------LatentSpace--------')
        print('loss_w_info: %s'%loss_w_info)
        print('loss_c_info: %s'%loss_c_info)

        it_d += 1
        writer.add_scalar('loss_mask_mse', loss_mask_info[0][0], global_step=it_d)
        writer.add_scalar('loss_mask_mse_mean', loss_mask_info[0][1], global_step=it_d)
        writer.add_scalar('loss_mask_mse_std', loss_mask_info[0][2], global_step=it_d)
        writer.add_scalar('loss_mask_kl', loss_mask_info[1], global_step=it_d)
        writer.add_scalar('loss_mask_cosine', loss_mask_info[2], global_step=it_d)
        writer.add_scalar('loss_mask_ssim', loss_mask_info[3], global_step=it_d)
        writer.add_scalar('loss_mask_lpips', loss_mask_info[4], global_step=it_d)

        writer.add_scalar('loss_grad_mse', loss_grad_info[0][0], global_step=it_d)
        writer.add_scalar('loss_grad_mse_mean', loss_grad_info[0][1], global_step=it_d)
        writer.add_scalar('loss_grad_mse_std', loss_grad_info[0][2], global_step=it_d)
        writer.add_scalar('loss_grad_kl', loss_grad_info[1], global_step=it_d)
        writer.add_scalar('loss_grad_cosine', loss_grad_info[2], global_step=it_d)
        writer.add_scalar('loss_grad_ssim', loss_grad_info[3], global_step=it_d)
        writer.add_scalar('loss_grad_lpips', loss_grad_info[4], global_step=it_d)

        writer.add_scalar('loss_imgs_mse', loss_imgs_info[0][0], global_step=it_d)
        writer.add_scalar('loss_imgs_mse_mean', loss_imgs_info[0][1], global_step=it_d)
        writer.add_scalar('loss_imgs_mse_std', loss_imgs_info[0][2], global_step=it_d)
        writer.add_scalar('loss_imgs_kl', loss_imgs_info[1], global_step=it_d)
        writer.add_scalar('loss_imgs_cosine', loss_imgs_info[2], global_step=it_d)
        writer.add_scalar('loss_imgs_ssim', loss_imgs_info[3], global_step=it_d)
        writer.add_scalar('loss_imgs_lpips', loss_imgs_info[4], global_step=it_d)

        writer.add_scalar('loss_Gcam', loss_Gcam_info[0][0], global_step=it_d)
        writer.add_scalar('loss_Gcam_mean', loss_Gcam_info[0][1], global_step=it_d)
        writer.add_scalar('loss_Gcam_std', loss_Gcam_info[0][2], global_step=it_d)
        writer.add_scalar('loss_Gcam_kl', loss_Gcam_info[1], global_step=it_d)
        writer.add_scalar('loss_Gcam_cosine', loss_Gcam_info[2], global_step=it_d)
        writer.add_scalar('loss_Gcam_ssim', loss_Gcam_info[3], global_step=it_d)
        writer.add_scalar('loss_Gcam_lpips', loss_Gcam_info[4], global_step=it_d)

        writer.add_scalar('loss_w_mse', loss_w_info[0][0], global_step=it_d)
        writer.add_scalar('loss_w_mse_mean', loss_w_info[0][1], global_step=it_d)
        writer.add_scalar('loss_w_mse_std', loss_w_info[0][2], global_step=it_d)
        writer.add_scalar('loss_w_kl', loss_w_info[1], global_step=it_d)
        writer.add_scalar('loss_w_cosine', loss_w_info[2], global_step=it_d)
        writer.add_scalar('loss_w_ssim', loss_w_info[3], global_step=it_d)
        writer.add_scalar('loss_w_lpips', loss_w_info[4], global_step=it_d)

        writer.add_scalar('loss_c_mse', loss_c_info[0][0], global_step=it_d)
        writer.add_scalar('loss_c_mse_mean', loss_c_info[0][1], global_step=it_d)
        writer.add_scalar('loss_c_mse_std', loss_c_info[0][2], global_step=it_d)
        writer.add_scalar('loss_c_kl', loss_c_info[1], global_step=it_d)
        writer.add_scalar('loss_c_cosine', loss_c_info[2], global_step=it_d)
        writer.add_scalar('loss_c_ssim', loss_c_info[3], global_step=it_d)
        writer.add_scalar('loss_c_lpips', loss_c_info[4], global_step=it_d)

        writer.add_scalars('Image_Space_MSE', {'loss_mask_mse':loss_mask_info[0][0],'loss_grad_mse':loss_grad_info[0][0],'loss_img_mse':loss_imgs_info[0][0]}, global_step=it_d)
        writer.add_scalars('Image_Space_KL', {'loss_mask_kl':loss_mask_info[1],'loss_grad_kl':loss_grad_info[1],'loss_imgs_kl':loss_imgs_info[1]}, global_step=it_d)
        writer.add_scalars('Image_Space_Cosine', {'loss_mask_cosine':loss_mask_info[2],'loss_grad_cosine':loss_grad_info[2],'loss_imgs_cosine':loss_imgs_info[2]}, global_step=it_d)
        writer.add_scalars('Image_Space_SSIM', {'loss_mask_ssim':loss_mask_info[3],'loss_grad_ssim':loss_grad_info[3],'loss_img_ssim':loss_imgs_info[3]}, global_step=it_d)
        writer.add_scalars('Image_Space_Lpips', {'loss_mask_lpips':loss_mask_info[4],'loss_grad_lpips':loss_grad_info[4],'loss_img_lpips':loss_imgs_info[4]}, global_step=it_d)
        writer.add_scalars('Latent Space W', {'loss_w_mse':loss_w_info[0][0],'loss_w_mse_mean':loss_w_info[0][1],'loss_w_mse_std':loss_w_info[0][2],'loss_w_kl':loss_w_info[1],'loss_w_cosine':loss_w_info[2]}, global_step=it_d)
        writer.add_scalars('Latent Space C', {'loss_c_mse':loss_c_info[0][0],'loss_c_mse_mean':loss_c_info[0][1],'loss_c_mse_std':loss_c_info[0][2],'loss_c_kl':loss_w_info[1],'loss_c_cosine':loss_w_info[2]}, global_step=it_d)

        if epoch % 100 == 0:
            n_row = batch_size
            test_img = torch.cat((imgs1[:n_row],imgs2[:n_row]))*0.5+0.5
            torchvision.utils.save_image(test_img, resultPath1_1+'/ep%d.png'%(epoch),nrow=n_row) # nrow=3
            heatmap=torch.cat((heatmap_1,heatmap_2))
            cam=torch.cat((cam_1,cam_2))
            grads = torch.cat((grad_1,grad_2))
            grads = grads.data.cpu().numpy() # [n,c,h,w]
            grads -= np.max(np.min(grads), 0)
            grads /= np.max(grads)
            torchvision.utils.save_image(torch.tensor(heatmap),resultPath_grad_cam+'/heatmap_%d.png'%(epoch),nrow=n_row)
            torchvision.utils.save_image(torch.tensor(cam),resultPath_grad_cam+'/cam_%d.png'%(epoch),nrow=n_row)
            torchvision.utils.save_image(torch.tensor(grads),resultPath_grad_cam+'/gb_%d.png'%(epoch),nrow=n_row)
            with open(resultPath+'/Loss.txt', 'a+') as f:
                print('i_'+str(epoch),file=f)
                print('[loss_imgs_mse[img,img_mean,img_std], loss_imgs_kl, loss_imgs_cosine, loss_imgs_ssim, loss_imgs_lpips]',file=f)
                print('---------ImageSpace--------',file=f)
                print('loss_mask_info: %s'%loss_mask_info,file=f)
                print('loss_grad_info: %s'%loss_grad_info,file=f)
                print('loss_imgs_info: %s'%loss_imgs_info,file=f)
                print('loss_Gcam_info: %s'%loss_Gcam_info,file=f)
                print('---------LatentSpace--------',file=f)
                print('loss_w_info: %s'%loss_w_info,file=f)
                print('loss_c_info: %s'%loss_c_info,file=f)
            if epoch % 5000 == 0:
                torch.save(E.state_dict(), resultPath1_2+'/E_model_ep%d.pth'%epoch)
                #torch.save(Gm.buffer1,resultPath1_2+'/center_tensor_ep%d.pt'%epoch)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='the training args')
    parser.add_argument('--epoch', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=0.0015)
    parser.add_argument('--beta_1', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--experiment_dir', default='none')
    parser.add_argument('--img_size',type=int, default=256)
    parser.add_argument('--img_channels', type=int, default=3)# RGB:3 ,L:1
    parser.add_argument('--z_dim', type=int, default=512) # BigGAN,z=128
    parser.add_argument('--mtype', type=int, default=1) # StyleGANv1=1, StyleGANv2=2, PGGAN=3, BigGAN=4
    args = parser.parse_args()

    if not os.path.exists('./result'): os.mkdir('./result')
    resultPath = args.experiment_dir
    if resultPath == 'none':
        resultPath = "./result/BigGAN-256"
        if not os.path.exists(resultPath): os.mkdir(resultPath)

    resultPath1_1 = resultPath+"/imgs"
    if not os.path.exists(resultPath1_1): os.mkdir(resultPath1_1)

    resultPath1_2 = resultPath+"/models"
    if not os.path.exists(resultPath1_2): os.mkdir(resultPath1_2)

    resultPath_grad_cam = resultPath+"/grad_cam"
    if not os.path.exists(resultPath_grad_cam): os.mkdir(resultPath_grad_cam)

    use_gpu = True
    device = torch.device("cuda" if use_gpu else "cpu")

    writer_path = os.path.join(resultPath, './summaries')
    if not os.path.exists(writer_path): os.mkdir(writer_path)
    writer = tensorboardX.SummaryWriter(writer_path)

    train(tensor_writer=writer, args= args)