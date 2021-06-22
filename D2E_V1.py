import os
import torch
import torchvision
import model.E.E_v3 as BE
from model.utils.custom_adam import LREQAdam
import metric.pytorch_ssim as pytorch_ssim
import lpips
import numpy as np
import tensorboardX
import argparse
from apex import amp
from model.stylegan1.net import Generator, Mapping #StyleGANv1
import model.stylegan2_generator as model_v2 #StyleGANv2
import model.pggan.pggan_generator as model_pggan #PGGAN
from model.biggan_generator import BigGAN #BigGAN

def one_hot(x, class_count=1000):
    # 第一构造一个[class_count, class_count]的对角线为1的向量
    # 第二保留label对应的行并返回
    return torch.eye(class_count)[x,:]

from scipy.stats import truncnorm
def truncated_noise_sample(batch_size=1, dim_z=128, truncation=1., seed=None):
    """ Create a truncated noise vector.
        Params:
            batch_size: batch size.
            dim_z: dimension of z
            truncation: truncation value to use
            seed: seed for the random generator
        Output:
            array of shape (batch_size, dim_z)
    """
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    return truncation * values

def set_seed(seed): #随机数设置
    np.random.seed(seed)
    #random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True

def space_loss(imgs1,imgs2,image_space=True,lpips_model=None):
    loss_mse = torch.nn.MSELoss()
    loss_kl = torch.nn.KLDivLoss()
    ssim_loss = pytorch_ssim.SSIM()
    loss_lpips = lpips_model

    imgs1 = imgs1.contiguous()
    imgs2 = imgs2.contiguous()

    loss_imgs_mse_1 = loss_mse(imgs1,imgs2)
    loss_imgs_mse_2 = loss_mse(imgs1.mean(),imgs2.mean())
    loss_imgs_mse_3 = loss_mse(imgs1.std(),imgs2.std())
    loss_imgs_mse = loss_imgs_mse_1 + loss_imgs_mse_2 + loss_imgs_mse_3

    imgs1_kl, imgs2_kl = torch.nn.functional.softmax(imgs1),torch.nn.functional.softmax(imgs2)
    loss_imgs_kl = loss_kl(torch.log(imgs2_kl),imgs1_kl) #D_kl(True=y1_imgs||Fake=y2_imgs)
    loss_imgs_kl = torch.where(torch.isnan(loss_imgs_kl),torch.full_like(loss_imgs_kl,0), loss_imgs_kl)
    loss_imgs_kl = torch.where(torch.isinf(loss_imgs_kl),torch.full_like(loss_imgs_kl,1), loss_imgs_kl)

    imgs1_cos = imgs1.view(-1)
    imgs2_cos = imgs2.view(-1)
    loss_imgs_cosine = 1 - imgs1_cos.dot(imgs2_cos)/(torch.sqrt(imgs1_cos.dot(imgs1_cos))*torch.sqrt(imgs2_cos.dot(imgs2_cos))) #[-1,1],-1:反向相反，1:方向相同

    if  image_space:
        ssim_value = pytorch_ssim.ssim(imgs1, imgs2) # while ssim_value<0.999:
        loss_imgs_ssim = 1-ssim_loss(imgs1, imgs2)
    else:
        loss_imgs_ssim = torch.tensor(0)

    if image_space:
        loss_imgs_lpips = loss_lpips(imgs1,imgs2).mean()
    else:
        loss_imgs_lpips = torch.tensor(0)

    loss_imgs = loss_imgs_mse + loss_imgs_kl + loss_imgs_cosine + loss_imgs_ssim + loss_imgs_lpips
    loss_info = [[loss_imgs_mse_1.item(),loss_imgs_mse_2.item(),loss_imgs_mse_3.item()], loss_imgs_kl.item(), loss_imgs_cosine.item(), loss_imgs_ssim.item(), loss_imgs_lpips.item()]
    return loss_imgs, loss_info


def train(tensor_writer = None, args = None):
    type = args.mtype

    if type == 1: # StyleGAN1

        Gs = Generator(startf=64, maxf=512, layer_count=7, latent_size=512, channels=3)
        Gs.load_state_dict(torch.load('./checkpoint/bedroom/bedrooms256_Gs_dict.pth'))

        Gm = Mapping(num_layers=14, mapping_layers=8, latent_size=512, dlatent_size=512, mapping_fmaps=512) #num_layers: 14->256 / 16->512 / 18->1024
        Gm.load_state_dict(torch.load('./checkpoint/bedroom/bedrooms256_Gm_dict.pth'))

        Gm.buffer1 = torch.load('./pre-model/bedroom/bedrooms256_tensor.pt')
        const1 = Gs.const
        layer_num = 14 # 14->256 / 16 -> 512  / 18->1024 
        layer_idx = torch.arange(layer_num)[np.newaxis, :, np.newaxis] # shape:[1,18,1], layer_idx = [0,1,2,3,4,5,6。。。，17]
        ones = torch.ones(layer_idx.shape, dtype=torch.float32) # shape:[1,18,1], ones = [1,1,1,1,1,1,1,1]
        coefs_ = torch.where(layer_idx < layer_num//2, 0.7 * ones, ones) # 18个变量前8个裁剪比例truncation_psi [0.7,0.7,...,1,1,1]

        Gs.eval()
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
        generator.eval()

    elif type == 3:  # PGGAN

        generator = model_pggan.PGGANGenerator(resolution=256).to(device)
        checkpoint = torch.load('./checkpoint/pggan_horse256.pth') #map_location='cpu'
        if 'generator_smooth' in checkpoint: #默认是这个
            generator.load_state_dict(checkpoint['generator_smooth'])
        else:
            generator.load_state_dict(checkpoint['generator'])
        const1 = torch.tensor(0)
        generator.eval()

    elif type == 4:

        cache_path = './checkpoint/biggan/256/G-256.pt'
        resolved_config_file = './checkpoint/biggan/256/biggan-deep-256-config.json'
        config = BigGANConfig.from_json_file(resolved_config_file)
        generator = BigGAN(config)
        generator.load_state_dict(torch.load(cache_path))
        generator.eval()

    else:
        print('error')
        return

    E = BE.BE(startf=64, maxf=512, layer_count=7, latent_size=512, channels=3)
    #E.load_state_dict(torch.load('/_yucheng/myStyle/myStyle-v1/EAE-car-cat/result/EB_cat_cosine_v2/E_model_ep80000.pth'))
    Gs.cuda()
    E.cuda()
    writer = tensor_writer

    E_optimizer = LREQAdam([{'params': E.parameters()},], lr=args.lr, betas=(args.beta_1, 0.99), weight_decay=0) 
    loss_lpips = lpips.LPIPS(net='vgg').to('cuda')

    if args.amp == True:
        E, E_optimizer = amp.initialize(E, E_optimizer, opt_level="O1") # 这里是“欧一”，不是“零一”

    batch_size = args.batch_size
    it_d = 0
    for epoch in range(0,args.epoch):
        set_seed(epoch%30000)
        z = torch.randn(batch_size, args.z_dim).cuda() #[32, 512]

        if type == 1:
            with torch.no_grad(): #这里需要生成图片和变量
                w1 = Gm(latents,coefs_m=coefs).to('cuda') #[batch_size,18,512]
                imgs1 = Gs.forward(w1,6) # 7->512 / 6->256
        elif type == 2:
            with torch.no_grad():
                result_all = generator(z, **synthesis_kwargs)
                imgs1 = result_all['image']
                w1 = result_all['wp']
        elif type == 3:
            with torch.no_grad(): #这里需要生成图片和变量
                w1 = z
                result_all = generator(w1)
                imgs1 = result_all['image']
        elif type == 4:
            z = truncated_noise_sample(truncation=synthesis_kwargs, batch_size=batch_size, seed=epoch%30000)
            #label = np.random.randint(1000,size=batch_size) # 生成标签
            flag = np.random.randint(1000)
            label = np.ones(batch_size)
            label = flag * label
            label = one_hot(label)
            w1 = torch.tensor(z, dtype=torch.float).cuda()
            conditions = torch.tensor(label, dtype=torch.float).cuda() # as label
            truncation = torch.tensor(synthesis_kwargs, dtype=torch.float).cuda()
            with torch.no_grad(): #这里需要生成图片和变量
                imgs1, const1 = G(w1, conditions, truncation) # const1 are conditional vectors in BigGAN

        if type != 4:
            const2,w2 = E(imgs1.cuda())
        else:
            const2,w2 = E(imgs1.cuda(), cond_vector)

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

#Latent-Vectors
## c
        loss_c, loss_c_info = space_loss(const1,const2,image_space = False)
        E_optimizer.zero_grad()
        if args.amp == True:
            with amp.scale_loss(loss_c, optimizer) as scaled_loss:
                scaled_loss.backward(retain_graph=True)
        else:
            loss_c.backward(retain_graph=True)
        E_optimizer.step()

## w
        loss_w, loss_w_info = space_loss(w1,w2,image_space = False)
        E_optimizer.zero_grad()
        if args.amp == True:
            with amp.scale_loss(loss_w, optimizer) as scaled_loss:
                scaled_loss.backward(retain_graph=True)
        else:
            loss_w.backward(retain_graph=True)
        E_optimizer.step()


#Image-Vectors 

##loss1 AT2
        imgs_small_1 = imgs1[:,:,imgs1.shape[2]//20:-imgs1.shape[2]//20,imgs1.shape[3]//20:-imgs1.shape[3]//20].clone() # w,h
        imgs_small_2 = imgs2[:,:,imgs2.shape[2]//20:-imgs2.shape[2]//20,imgs2.shape[3]//20:-imgs2.shape[3]//20].clone()
        loss_small, loss_small_info = space_loss(imgs_small_1,imgs_small_2,lpips_model=loss_lpips)
        E_optimizer.zero_grad()
        if args.amp == True:
            with amp.scale_loss(loss_small, optimizer) as scaled_loss:
                scaled_loss.backward(retain_graph=True)
        else:
            loss_small.backward(retain_graph=True)
        E_optimizer.step()


#loss2 AT1
        imgs_medium_1 = imgs1[:,:,imgs1.shape[2]//10:-imgs1.shape[2]//10,imgs1.shape[3]//10:-imgs1.shape[3]//10].clone()
        imgs_medium_2 = imgs2[:,:,imgs2.shape[2]//10:-imgs2.shape[2]//10,imgs2.shape[3]//10:-imgs2.shape[3]//10].clone()
        loss_medium, loss_medium_info = space_loss(imgs_medium_1,imgs_medium_2,lpips_model=loss_lpips)
        E_optimizer.zero_grad()
        if args.amp == True:
            with amp.scale_loss(loss_medium, optimizer) as scaled_loss:
                scaled_loss.backward(retain_graph=True)
        else:
            loss_medium.backward(retain_graph=True)
        E_optimizer.step()

#loss3 Images
        loss_imgs, loss_imgs_info = space_loss(imgs1,imgs2,lpips_model=loss_lpips)
        E_optimizer.zero_grad()
        if args.amp == True:
            with amp.scale_loss(loss_imgs, optimizer) as scaled_loss:
                scaled_loss.backward(retain_graph=True)
        else:
            loss_imgs.backward(retain_graph=True)
        E_optimizer.step()

        print('i_'+str(epoch))
        print('[loss_imgs_mse[img,img_mean,img_std], loss_imgs_ssim, loss_imgs_cosine, loss_kl_imgs, loss_imgs_lpips]')
        print('---------ImageSpace--------')
        print('loss_small_info: %s'%loss_small_info)
        print('loss_medium_info: %s'%loss_medium_info)
        print('loss_imgs_info: %s'%loss_imgs_info)
        print('---------LatentSpace--------')
        print('loss_w_info: %s'%loss_w_info)
        print('loss_c_info: %s'%loss_c_info)

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

        writer.add_scalar('loss_c_mse', loss_c_info[0][0], global_step=it_d)
        writer.add_scalar('loss_c_mse_mean', loss_c_info[0][1], global_step=it_d)
        writer.add_scalar('loss_c_mse_std', loss_c_info[0][2], global_step=it_d)
        writer.add_scalar('loss_c_kl', loss_c_info[1], global_step=it_d)
        writer.add_scalar('loss_c_cosine', loss_c_info[2], global_step=it_d)
        writer.add_scalar('loss_c_ssim', loss_c_info[3], global_step=it_d)
        writer.add_scalar('loss_c_lpips', loss_c_info[4], global_step=it_d)

        writer.add_scalars('Image_Space_MSE', {'loss_small_mse':loss_small_info[0][0],'loss_medium_mse':loss_medium_info[0][0],'loss_img_mse':loss_imgs_info[0][0]}, global_step=it_d)
        writer.add_scalars('Image_Space_KL', {'loss_small_kl':loss_small_info[1],'loss_medium_kl':loss_medium_info[1],'loss_imgs_kl':loss_imgs_info[1]}, global_step=it_d)
        writer.add_scalars('Image_Space_Cosine', {'loss_samll_cosine':loss_small_info[2],'loss_medium_cosine':loss_medium_info[2],'loss_imgs_cosine':loss_imgs_info[2]}, global_step=it_d)
        writer.add_scalars('Image_Space_SSIM', {'loss_small_ssim':loss_small_info[3],'loss_medium_ssim':loss_medium_info[3],'loss_img_ssim':loss_imgs_info[3]}, global_step=it_d)
        writer.add_scalars('Image_Space_Lpips', {'loss_small_lpips':loss_small_info[4],'loss_medium_lpips':loss_medium_info[4],'loss_img_lpips':loss_imgs_info[4]}, global_step=it_d)
        writer.add_scalars('Latent Space W', {'loss_w_mse':loss_w_info[0][0],'loss_w_mse_mean':loss_w_info[0][1],'loss_w_mse_std':loss_w_info[0][2],'loss_w_kl':loss_w_info[1],'loss_w_cosine':loss_w_info[2]}, global_step=it_d)
        writer.add_scalars('Latent Space C', {'loss_c_mse':loss_c_info[0][0],'loss_c_mse_mean':loss_c_info[0][1],'loss_c_mse_std':loss_c_info[0][2],'loss_c_kl':loss_c_info[1],'loss_c_cosine':loss_c_info[2]}, global_step=it_d)


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
    parser.add_argument('--z_dim', type=int, default=512)
    parser.add_argument('--mtype', type=int, default=1) # StyleGANv1=1, StyleGANv2=2, PGGAN=3, BigGAN=4
    parser.add_argument('--amp', type=bool, default=True) 
    args = parser.parse_args()

    if not os.path.exists('./result'): os.mkdir('./result')
    resultPath = args.experiment_dir
    if resultPath == 'none':
        resultPath = "./result/StyleGANv1_bedroom"
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
