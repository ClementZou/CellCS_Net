import sys

from model import ISTANetplus
from utils import RandomDataset, weights_init_normal
import utils
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import platform
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
import time

parser = ArgumentParser(description='ISTA-Net-plus')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=200, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=9, help='phase number of ISTA-Net-plus')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--y_size', type=int, default=5000, help='the size of y')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--generate_time', type=str, default='default',
                    help='to recognize different tensorBoard and models')

parser.add_argument('--model_name', type=str, default='ISTANetplus', help='name of trained or pre-trained model')
parser.add_argument('--training_data_dir', type=str, default='', help='name of training data')
parser.add_argument('--tensor_board', type=bool, default=False, help='')
parser.add_argument('--channel_number', type=int, default=32, help='channel number of conv2d')
parser.add_argument('--training_type', type=str, default='ref', help='training type:ref or sim')

args = parser.parse_args()

# generate_time = args.generate_time
tensorBoard_flag = args.tensor_board
start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
y_size = args.y_size
gpu_list = args.gpu_list
Nf = args.channel_number
generate_time = args.generate_time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_output = 16384
batch_size = 16


if tensorBoard_flag:
    tensorBoardPath = './tensorBoard/%s_%s_%s_lr_%.5f_Nf_%d' % (
        args.training_data_dir, args.model_name, args.training_type, learning_rate, Nf)
    if not os.path.exists(tensorBoardPath):
        os.makedirs(tensorBoardPath)
    writer = SummaryWriter('%s/' % tensorBoardPath)

if args.training_type == 'ref':
    Phi, training_labels, Qinit = utils.data_loader_ref(args.training_data_dir, y_size, device)
elif args.training_type == 'sim':
    Phi, training_labels, Qinit = utils.data_loader_sim(args.training_data_dir, y_size, device)
else:
    print('training_type should be \'sim\' for simulate or \'ref\' for reference')
    sys.exit()
nrtrain = training_labels.shape[0]  # number of training blocks
rand_loader = DataLoader(dataset=RandomDataset(training_labels, nrtrain), batch_size=batch_size, num_workers=2,
                         shuffle=True)
if args.model_name == 'ISTANetplus':
    print('ISTANetplus')
    model = ISTANetplus(layer_num, Nf)
else:
    print('please choose an exist model')
    sys.exit()
model = nn.DataParallel(model)
model = model.to(device)

# 模型参数权重初始化
model.apply(weights_init_normal)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = './models/%s/%s_%s_lr_%.5f_Nf_%d' % (
    args.training_data_dir, args.model_name, args.training_type, learning_rate, Nf)


log_file_name = "./log/%s/%s_%s_lr_%.5f_Nf_%d_layerNum_%d_ySize_%d.txt" % (
    args.training_data_dir, args.model_name, args.training_type, learning_rate, Nf, layer_num, y_size)

os.makedirs(model_dir, exist_ok=True)
os.makedirs("./log/%s" % args.training_data_dir, exist_ok=True)

if start_epoch > 0:
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('%s/net_params_%d.pkl' % (pre_model_dir, start_epoch)))

# Training loop
last_loss = -1
j = 0
for epoch_i in range(start_epoch + 1, end_epoch + 1):
    for data in rand_loader:
        if args.training_type == 'ref':
            batch_x = data[:, :16384]
            batch_x = batch_x.to(device)
            phiX = data[:, 16384:]
        else:
            batch_x = data.to(device)
            phiX = torch.mm(batch_x, torch.transpose(Phi, 0, 1))
            # # 增加噪声
            # noise = torch.rand(batch_size, y_size)
            # noise = noise.to(device)
            # noise = ((noise + 0.0131) * 0.0581) * (phiX.max() - phiX.min())
            # phiX = phiX + noise
        for i in range(batch_x.shape[0]):
            batch_x[i] = (batch_x[i] - batch_x[i].min()) / (batch_x[i].max() - batch_x[i].min())
            phiX[i] = (phiX[i] - phiX[i].min()) * 2 / (phiX[i].max() - phiX[i].min()) - 1

        phiX = phiX.to(device)
        [x_output, loss_layers_sym] = model(phiX, Phi, Qinit)

        # Compute and print loss
        Prediction_value = x_output.cpu().data.numpy()
        X_rec = np.reshape(Prediction_value, (-1, 128, 128))
        X_ori = np.reshape(batch_x.cpu().data.numpy(), (-1, 128, 128))
        IMG = np.concatenate((np.clip(X_ori[0], 0, 1), np.clip(X_rec[0], 0, 1)), axis=1)

        loss_all = torch.mean(torch.pow(x_output - batch_x, 2))
        if loss_all / last_loss <= 10:
            j = j + 1
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            last_loss = loss_all

        if tensorBoard_flag:
            writer.add_image('IMG', IMG, global_step=j, dataformats='HW')
            writer.add_scalar('loss_all', loss_all, global_step=j)
        output_data = "[%02d/%02d] Total Loss: %.4f\n" % (
            epoch_i, end_epoch, loss_all.item())
        print(output_data)

    output_file = open(log_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    if epoch_i % 5 == 0:
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters
if tensorBoard_flag:
    writer.close()
