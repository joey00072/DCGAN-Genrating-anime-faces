#imports
import gzip,pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


#dataset url : https://www.kaggle.com/splcher/animefacedataset
#change path to unzip foler of images
PATH="../dataset/"


#Constants & Hyperparam
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-5  # could also use two lrs, one for gen and one for D
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
NOISE_DIM = 100
NUM_EPOCHS = 200
FEATURES_DISC = 64
FEATURES_GEN = 64



#prepossessing
    
class Prepro:
    def __init__(self):
         for item in tqdm(os.listdir(PATH+"/images")):
            img = cv2.imread(f"{PATH}/images/{item}")
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(IMAGE_SIZE,IMAGE_SIZE))
            np.save(f"anime_images/{item[:-4]}.npy",img)

if not os.path.exists("anime_images") :
    os.mkdir("anime_images")
    Prepro()



class Dataset_Anime(torch.utils.data.Dataset):
    def  __init__(self):
        super(Dataset_Anime, self).__init__()
        self.data = os.listdir("./anime_images")
        self.K  = 3
        self.len = len(self.data)
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        try:
            img = np.load(f"anime_images/{self.data[index%len(self.data)]}",allow_pickle=True)
        except Exception as e:
            img = np.load(f"anime_images/{self.data[0]}",allow_pickle=True)
            print(str(e))
        img = torch.from_numpy(img/255).float().permute(2,0,1)
        return img,img
dataset=Dataset_Anime()



class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(
                channels_img, 
                features_d, 
                kernel_size=4, 
                stride=2, 
                padding=1
            ),
            nn.LeakyReLU(0.2),
            self._block(features_d    , features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self._block(channels_noise, features_g * 16, 4, 1, 0),  
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  
            nn.ConvTranspose2d(
                features_g * 2, 
                channels_img, 
                kernel_size=4, 
                stride=2, 
                padding=1
            ),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)



def initialize_weights(model):
    # init weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)




dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
G = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
D = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(G)
initialize_weights(D)




opt_G = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

G.train()
D.train()




for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        
        noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
        fake = G(noise)

        D_real = D(real).reshape(-1)
        loss_D_real = criterion(D_real, torch.ones_like(D_real))
        D_fake = D(fake.detach()).reshape(-1)
        loss_D_fake = criterion(D_fake, torch.zeros_like(D_fake))
        loss_D = (loss_D_real + loss_D_fake) / 2
        D.zero_grad()
        loss_D.backward()
        opt_D.step()
        
        output = D(fake).reshape(-1)
        loss_G = criterion(output, torch.ones_like(output))
        G.zero_grad()
        loss_G.backward()
        opt_G.step()

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {loss_D:.4f}, loss G: {loss_G:.4f}"
            )

            with torch.no_grad():
                fake = G(fixed_noise)
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                

            step += 1




torch.save(G.state_dict(),'anime_gen.pth')