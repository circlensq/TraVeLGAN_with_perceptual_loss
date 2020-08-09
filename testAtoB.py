from data import get_datasets
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

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--hparams", type=str, default='cifar', help="hparams config file")
parser.add_argument('--project_name', required=False, default='photo2emoji',  help='project name')
parser.add_argument('--input_size', type=int, default=64, help='input size')
opts = parser.parse_args()

print(opts)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.isdir(os.path.join(opts.project_name + '_results', 'Test_AtoB_model_500')):
    os.makedirs(os.path.join(opts.project_name + '_results', 'Test_AtoB_model_500'))

# Generate seed
opts.manualSeed = random.randint(1, 10000)
random.seed(opts.manualSeed)
torch.manual_seed(opts.manualSeed)
print('Random Seed: ', opts.manualSeed)

# Transform dataset
src_transform = transforms.Compose([
        transforms.Resize((opts.input_size, opts.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

print('Loading data..')
hparams = load_json('./configs', opts.hparams)
test_src = data_load(os.path.join('./dataset/', 'CelebA/'), 'test', src_transform, batch_size=1, shuffle=False, drop_last=True)

model = TravelGAN(hparams['model'], device=device)
if  hparams['saved_model'] :
    print('saved model : ', hparams['saved_model'])
    model.resume(hparams['saved_model'])

with torch.no_grad():
    model.eval()
    for n, (x_a, _) in enumerate(test_src):
        # Loading on device
        x_a = x_a.to(device)

        x_ab =  model.transformToCartoon(x_a)
        x_ab = x_ab.detach()

        result = torch.cat((x_a[0], x_ab[0]), 2)
        path = os.path.join(opts.project_name + '_results', 'Test_AtoB_model_500/' + str(n+1) + '.png')
        plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)


