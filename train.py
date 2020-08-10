from trainer import TravelGAN
from torch.utils.data.dataloader import DataLoader
from utils import get_device, load_json, data_load
import argparse
from statistics import mean
import torch
import os
import matplotlib.pyplot as plt
import random
import torchvision.transforms as transforms

from edge_promoting import edge_promoting

parser = argparse.ArgumentParser()
parser.add_argument("--log", type=str, default='./log_photo2emoji/', help="name of log folder")
parser.add_argument("-p", "--hparams", type=str, default='config', help="hparams config file")
parser.add_argument('--project_name', required=False, default='photo2emoji',  help='project name')
parser.add_argument('--input_size', type=int, default=64, help='input size')

opts = parser.parse_args()

print(opts)

# Get CUDA/CPU device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not os.path.isdir(os.path.join(opts.project_name + '_results', 'Transfer')):
    os.makedirs(os.path.join(opts.project_name + '_results', 'Transfer'))

if not os.path.isdir(opts.log):
    os.makedirs(os.path.join(opts.log))

# Generate seed
opts.manualSeed = random.randint(1, 10000)
random.seed(opts.manualSeed)
torch.manual_seed(opts.manualSeed)
print('Random Seed: ', opts.manualSeed)

# Transform dataset
src_transform = transforms.Compose([
        transforms.Resize((opts.input_size, 2*opts.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

tgt_transform = transforms.Compose([
        transforms.Resize((opts.input_size, 2*opts.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
   

print('Loading data..')
hparams = load_json('./configs', opts.hparams)
if not os.path.isdir(os.path.join('./dataset/', 'CelebA/', 'trainA_pair')):
    print('Domain A edge-promoting start!!')
    edge_promoting(os.path.join('./dataset/', 'CelebA/train/'), os.path.join('./dataset/', 'CelebA/trainA_pair/'))
else:
    print('Domain A edge-promoting start!!')

if not os.path.isdir(os.path.join('./dataset/', 'Bitmoji/trainB_pair/')):
    print('Domain B edge-promoting start!!')
    edge_promoting(os.path.join('./dataset/', 'Bitmoji/'), os.path.join('./dataset/', 'Bitmoji/trainB_pair/'))
else:
    print('Domain B edge-promoting start!!')


loading = hparams['loading']
train_loader_src = data_load(os.path.join('./dataset/', 'CelebA/'), 'trainA_pair', src_transform, batch_size=loading['batch_size'], shuffle=loading['shuffle'], drop_last=True)
train_loader_tgt = data_load(os.path.join('./dataset/', 'Bitmoji/'), 'trainB_pair', tgt_transform, batch_size=loading['batch_size'], shuffle=loading['shuffle'], drop_last=True)
test_loader_src = data_load(os.path.join('./dataset/', 'CelebA/'), 'test', src_transform, batch_size=loading['batch_size'], shuffle=loading['shuffle'], drop_last=True)

model = TravelGAN(hparams['model'], device=device)
if  hparams['saved_model'] or opts.log:
    if hparams['saved_model']:
        print('saved model : ', hparams['saved_model'])
        model.resume(hparams['saved_model'])
    else :
        print('saved model : ', opts.log)
        model.resume(opts.log)


print('Start training..')
for epoch in range(hparams['n_epochs']):
    # Run one epoch
    torch.cuda.empty_cache()
    dis_losses, gen_losses, siam_losses = [], [], []
    for (x_a, _ ), (x_b,_) in zip(train_loader_src, train_loader_tgt):
        x_a_edge = x_a[:, :, :, opts.input_size: ]
        x_a = x_a[:, :, :, :opts.input_size]
      
        x_b_edge = x_b[:, :, :, opts.input_size: ]
        x_b = x_b[:, :, :, :opts.input_size]

        x_a, x_a_edge = x_a.to(device), x_a_edge.to(device)
        x_b, x_b_edge = x_b.to(device), x_b_edge.to(device)

        # Calculate losses and update weights
        dis_loss = model.dis_update(x_a, x_b, x_a_edge, x_b_edge)
        x_ab, x_ba, gen_loss, siam_loss = model.gen_update(x_a, x_b)

        dis_losses.append(dis_loss)
        gen_losses.append(gen_loss)
        siam_losses.append(siam_loss)

    # Logging losses
    dis_loss, gen_loss, siam_loss = mean(dis_losses), mean(gen_losses), mean(siam_losses)

    if epoch % 2 == 1 or epoch == hparams['n_epochs'] - 1:
        n = 0
        for (x_a, _ ), (x_b,_) in zip(train_loader_src, train_loader_tgt):
      
            x_a = x_a[:, :, :, :opts.input_size]
            x_b = x_b[:, :, :, :opts.input_size]

            x_a = x_a.to(device)
            x_b = x_b.to(device)

            x_ab, x_ba, gen_loss, siam_loss = model.gen_update(x_a, x_b)
            x_ab = x_ab.detach()
            x_ba = x_ba.detach()
            result_1 = torch.cat((x_a[0], x_ab[0]), 2)
            result_2 = torch.cat((x_b[0], x_ba[0]), 2)
            path_1 = os.path.join(opts.project_name + '_results', 'Transfer', str(epoch+1) + '_epoch_' + opts.project_name + '_train_AtoB' + '.png')
            path_2 = os.path.join(opts.project_name + '_results', 'Transfer', str(epoch+1) + '_epoch_' + opts.project_name + '_train_BtoA' + '.png')
            plt.imsave(path_1, (result_1.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
            plt.imsave(path_2, (result_2.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
           
            n += 1
            break
        print('Epoch: %d ,Gen loss: %.2f, Disc loss: %.2f. Siam loss: %.2f,' % (epoch, gen_loss, dis_loss, siam_loss))
    # Saving model every n_save_steps epochs
    if (epoch + 1) % hparams['n_save_steps'] == 0:
        model.save(opts.log, epoch)
