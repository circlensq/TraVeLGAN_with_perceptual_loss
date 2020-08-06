import torch
import torch.nn as nn
from networks.default import Generator, Discriminator, Siamese, VGG19
from losses import AdversarialLoss, TravelLoss, MarginContrastiveLoss
from losses import compute_gp, tv_loss_reg
from torch.optim import Adam
from utils import initialize_weights
import torchvision.models as models

import os
import copy

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

class TravelGAN(nn.Module):
    def __init__(self, hparams, device="cpu", ):
        super(TravelGAN, self).__init__()
        # Parameters
        self.hparams = hparams
        self.device = device
        self.VGG = VGG19(feature_mode=True).to(self.device)
    
        # Modules
        self.gen_ab = Generator(**hparams["gen"])
        self.gen_ab.unet_block_1.register_forward_hook(get_activation('block1'))
        self.gen_ab.unet_block_2.register_forward_hook(get_activation('block2'))
        self.gen_ab.unet_block_3.register_forward_hook(get_activation('block3'))

        self.gen_ab_encode = Generator(**hparams["gen"])
        self.gen_ab_encode.unet_block_1.register_forward_hook(get_activation('block1'))
        self.gen_ab_encode.unet_block_2.register_forward_hook(get_activation('block2'))
        self.gen_ab_encode.unet_block_3.register_forward_hook(get_activation('block3'))

        self.gen_ba = Generator(**hparams["gen"])
        self.gen_ba.unet_block_1.register_forward_hook(get_activation('block1'))
        self.gen_ba.unet_block_2.register_forward_hook(get_activation('block2'))
        self.gen_ba.unet_block_3.register_forward_hook(get_activation('block3'))
      
        self.gen_ba_encode = Generator(**hparams["gen"])
        self.gen_ba_encode.unet_block_1.register_forward_hook(get_activation('block1'))
        self.gen_ba_encode.unet_block_2.register_forward_hook(get_activation('block2'))
        self.gen_ba_encode.unet_block_3.register_forward_hook(get_activation('block3'))
      
        self.dis_a = Discriminator(**hparams["dis"])
        self.dis_b = Discriminator(**hparams["dis"])
        self.siam = Siamese(**hparams["siam"])

        # Loss coefficients
        self.lambda_adv = hparams["lambda_adv"]
        self.lambda_travel = hparams["lambda_travel"]
        self.lambda_margin = hparams["lambda_margin"]
        self.margin = hparams["margin"]
        self.lambda_gp = hparams["lambda_gp"]
        self.lambda_con = hparams["lambda_con"]
        self.lambda_sem = hparams["lambda_sem"]
        self.type = hparams["type"]

        # Learning rates
        self.lr_dis = hparams["lr_dis"]
        self.lr_gen = hparams["lr_gen"]

        # Optimizers
        dis_params = list(self.dis_a.parameters()) + \
            list(self.dis_b.parameters())
        gen_params = list(self.gen_ab.parameters()) + \
            list(self.gen_ba.parameters()) + list(self.siam.parameters())
        self.dis_optim = Adam([p for p in dis_params],
                              lr=self.lr_dis, betas=(0.5, 0.9))
        self.gen_optim = Adam([p for p in gen_params],
                              lr=self.lr_gen, betas=(0.5, 0.9))

        # Losses
        self.adv_loss = AdversarialLoss(self.type, device)
        if self.type == "wgangp":
            self.gp = compute_gp
        self.travel_loss = TravelLoss()
        self.margin_loss = MarginContrastiveLoss(self.margin)

        self.total_variation_loss = tv_loss_reg
        # Initialization
        self.apply(initialize_weights)
        self.set_to(device)

    def forward(self, x_a, x_b):
        self.eval()
        return self.gen_ab(x_a), self.gen_ba(x_b)
    
    def transformToCartoon(self, x_a):
        self.eval()
        return self.gen_ab(x_a)

    def transformToReal(self, x_b):
        self.eval()
        return self.gen_ba(x_b)
    
    def dis_update(self, x_a, x_b, x_a_edge, x_b_edge):
        self.dis_optim.zero_grad()
        x_ab = self.gen_ab(x_a).detach()
        x_ba = self.gen_ba(x_b).detach()
       
        adv_loss = self.adv_loss(self.dis_a(x_a), True) + \
            self.adv_loss(self.dis_b(x_b), True) + \
            self.adv_loss(self.dis_a(x_a_edge), False) + \
            self.adv_loss(self.dis_b(x_b_edge), False) + \
            self.adv_loss(self.dis_b(x_ab), False) + \
            self.adv_loss(self.dis_a(x_ba), False)
        dis_loss = self.lambda_adv * adv_loss
        if self.type == "wgangp":
            gp = self.gp(self.dis_a, x_a, x_ba) + \
                self.gp(self.dis_b, x_b, x_ab)
            dis_loss += self.lambda_gp * gp
        dis_loss.backward()
        self.dis_optim.step()
        return dis_loss.item()

    def get_vgg19_model(cnn, image, conv_layers):
        cnn_2 = copy.deepcopy(cnn)

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        i = 0  # increment every time we see a conv
        for layer in cnn_2.children():
            print('layer',layer)
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

        model = model[:(i + 1)]

        return model

    def gen_update(self, x_a, x_b):
        self.gen_optim.zero_grad()
        x_ab = self.gen_ab(x_a)
        x_ab_encode = self.gen_ab_encode(x_ab)
        
        x_ba = self.gen_ba(x_b)
        x_ba_encode = self.gen_ba_encode(x_ba)
        
        L1_loss = nn.L1Loss()
        L2_loss = nn.MSELoss()
        
        x_a_feature = self.VGG((x_a + 1) / 2)
       
        x_ab_feature = self.VGG((x_ab + 1) / 2)
        preserved_con_loss_ab = self.lambda_con * L2_loss(x_ab_feature[0], x_a_feature[0].detach())

        x_ba_feature = self.VGG((x_ba + 1) / 2)
        x_b_feature = self.VGG((x_b + 1) / 2)
        preserved_con_loss_ba = self.lambda_con * L2_loss(x_ba_feature[0], x_b_feature[0].detach())

        adv_loss = self.adv_loss(self.dis_b(x_ab), True) + \
            self.adv_loss(self.dis_a(x_ba), True)
        travel_loss = self.travel_loss(x_a, x_ab, self.siam) + \
            self.travel_loss(x_b, x_ba, self.siam)
        margin_loss = self.margin_loss(x_a, self.siam) + self.margin_loss(x_b, self.siam)
        
        con_loss = preserved_con_loss_ab + preserved_con_loss_ba
        
        total_variation_loss = self.total_variation_loss(x_ab) + self.total_variation_loss(x_ba)

        siam_loss = self.lambda_margin * margin_loss + \
            self.lambda_travel * travel_loss 

        gen_loss = self.lambda_adv * adv_loss + \
            self.lambda_travel * travel_loss + \
            total_variation_loss + \
            con_loss
        
        gen_loss.backward(retain_graph=True)
        siam_loss.backward(retain_graph=True)
        self.gen_optim.step()
        
        return x_ab, x_ba, gen_loss.item(), siam_loss.item()

    def resume(self, file):
        state_dict = torch.load(file, map_location=self.device)
        self.load_state_dict(state_dict)

    def save(self, checkpoint_dir, epoch):
        file = 'model_{}.pt'.format(epoch + 1)
        file = os.path.join(checkpoint_dir, file)
        torch.save(self.state_dict(), file)

    def set_to(self, device):
        self.device = device
        self.to(device)
        print("Model loaded on device : {}".format(device))
