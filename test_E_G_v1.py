#test_E_v1.py
import math
import torch
import torchvision
from model.stylegan1.net import Generator, Mapping

def train():

    model_path = './checkpoint/stylegan_v1/ffhq1024/'
    Gs = Generator(startf=16, maxf=512, layer_count=int(math.log(1024,2)-1), latent_size=512, channels=3)
    Gs.load_state_dict(torch.load(model_path+'Gs_dict.pth'))
    Gs.to(device)

    w = torch.load('msv-rec30-w.pt')
    imgs = []
    for i,j in enumerate(w):
        w1 = j.view(1,18,512)
        with torch.no_grad():
            img1 = Gs.forward(w1,int(math.log(1024,2)-2))
            torchvision.utils.save_image(img1*0.5+0.5,'./img-%d.png'%i)
            imgs.append(img1[0])

    imgs_tensor= torch.stack(imgs)
    print(imgs_tensor.shape)
    torch.save(imgs_tensor,'./msv_30rec-img.pt')
    torchvision.utils.save_image(imgs_tensor*0.5+0.5,'./msv-imgs.png',nrow=5)

if __name__ == "__main__":

    use_gpu = True
    device = torch.device("cuda" if use_gpu else "cpu")
    train()