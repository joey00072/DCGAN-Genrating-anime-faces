import pygame
import numpy as np
import cv2
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from threading import Thread


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 50)
BLUE = (50, 50, 255)
GREY = (200, 200, 200)
ORANGE = (200, 100, 50)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
TRANS = (1, 1, 1)

WIDTH = 100*6
HIGHT = 100*6
pygame.init()
screen = pygame.display.set_mode((WIDTH, HIGHT))
font = pygame.font.SysFont("Verdana", 12)
clock = pygame.time.Clock()



class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
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
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


device = torch.device('cuda')
G = Generator(100,3,64).to(device)
G.load_state_dict(torch.load("anime_gen.pth"))
G.eval()
no_of_img = 5
LATENT = torch.randn(no_of_img*no_of_img, 100, 1, 1).to(device)


with torch.no_grad():
    out = G(LATENT).to(device)

    out= out.permute(0,3,2,1).detach().cpu().numpy()
    out = out+abs(out.min())
    out = out/out.max()
    out= out*255


# noise = torch.randn(1, 100, 1, 1)/10



mute = torch.randn(no_of_img*no_of_img, 100, 1, 1).to(device)/10

def random_mutate(idx):
    global LATENT,mute

    if idx==0:
        mute = torch.randn(no_of_img*no_of_img, 100, 1, 1,device=device)/10

    LATENT = LATENT+mute
    K=0.99
    LATENT[LATENT>K]=K
    LATENT[LATENT<-K]=-K


def fpass(LATENT,G):
    out = G(LATENT)
    # print(out.min())
    out= out.permute(0,3,2,1).detach().cpu().numpy()
    # out = out+0.5
    out = out +0.3
    out = out/out.max()
    # print(out.max())

    out= out*230
    return out


def display_grid(idx):
    random_mutate(idx)

    out = fpass(LATENT,G)
    for x in range(no_of_img):
        for y in range(no_of_img):
            idx = x*no_of_img+y
            img = out[idx]
            surface = pygame.surfarray.make_surface(img)
            surface = pygame.transform.scale(surface, (120, 120))
            screen.blit(surface, (x*120, y*120))

idx=0

with torch.no_grad():
    RUN=True
    while RUN:
        screen.fill(BLACK)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()


        display_grid(idx)
        idx+=1
        idx%=30
        pygame.display.update()
        clock.tick(20)
    pygame.quit()