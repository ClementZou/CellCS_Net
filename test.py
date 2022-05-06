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
from argparse import ArgumentParser
from skimage.measure import compare_ssim as ssim
import cv2

parser = ArgumentParser(description='ISTA-Net-plus')

parser.add_argument('--epoch_num', type=int, default=160, help='epoch number of model')
parser.add_argument('--layer_num', type=int, default=12, help='phase number of ISTA-Net-plus')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--y_size', type=int, default=5000, help='the size of y')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--generate_time', type=str, default='default',
                    help='to recognize different tensorBoard and models')

parser.add_argument('--model_name', type=str, default='ISTANetplus', help='name of trained or pre-trained model')
parser.add_argument('--training_data_dir', type=str, default='', help='name of training data')
parser.add_argument('--test_data_dir', type=str, default='', help='name of test data')
parser.add_argument('--channel_number', type=int, default=32, help='channel number of conv2d')
parser.add_argument('--training_type', type=str, default='ref', help='training type:ref or sim')

args = parser.parse_args()

epoch_num = args.epoch_num
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

if args.training_type == 'ref':
    Phi, training_labels, Qinit = utils.data_loader_ref(args.test_data_dir, y_size, device)
elif args.training_type == 'sim':
    Phi, training_labels, Qinit = utils.data_loader_sim(args.test_data_dir, y_size, device)
else:
    print('training_type should be \'sim\' for simulate or \'ref\' for reference')
    sys.exit()

# select the data for test here
test_data = training_labels[:50, :]

ImgNum = test_data.shape[0]  # number of test

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
# Load pre-trained model with epoch number
model.load_state_dict(torch.load('%s/net_params_%d.pkl' % (model_dir, epoch_num)))

result_dir = './result/%s/%s_%s_lr_%.5f_Nf_%d/%s' % (
    args.training_data_dir, args.model_name, args.training_type, learning_rate, Nf, args.test_data_dir)

log_file_name = "%s/log.txt" % result_dir

os.makedirs(result_dir, exist_ok=True)

PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)

# Test loop
with torch.no_grad():
    for img_no in range(ImgNum):
        batch_x = torch.from_numpy(test_data[img_no, :16384])
        batch_x = torch.unsqueeze(batch_x, 0)
        batch_x = batch_x.type(torch.FloatTensor)
        batch_x = batch_x.to(device)

        for i in range(batch_x.shape[0]):
            batch_x[i] = (batch_x[i] - batch_x[i].min()) / (
                        batch_x[i].max() - batch_x[i].min())
        if args.training_type == 'ref':
            batch_y = torch.from_numpy(test_data[img_no,16384:])
            batch_y = torch.unsqueeze(batch_y, 0)
            batch_y = batch_y.type(torch.FloatTensor)
        else:
            batch_y = torch.mm(batch_x, torch.transpose(Phi, 0, 1))
        for i in range(batch_y.shape[0]):
            batch_y[i] = (batch_y[i] - batch_y[i].min()) * 2 / (batch_y[i].max() - batch_y[i].min()) - 1
        batch_y = batch_y.to(device)

        [x_output, loss_layers_sym] = model(batch_y, Phi, Qinit)

        Prediction_value = x_output.cpu().data.numpy()
        X_rec = np.reshape(np.clip(Prediction_value, 0, 1), (128, 128))
        X_ori = np.reshape(test_data[img_no, :16384], (128, 128))
        X_ori = (X_ori - X_ori.min()) / (X_ori.max() - X_ori.min())

        rec_PSNR = utils.psnr(X_rec, X_ori.astype(np.float32))
        rec_SSIM = ssim(X_rec * 255, X_ori.astype(np.float32) * 255, data_range=255)

        PSNR_All[0, img_no] = rec_PSNR
        SSIM_All[0, img_no] = rec_SSIM

        cv2.imwrite('%s/%d_result.png' % (result_dir, img_no), X_rec * 255)
        cv2.imwrite('%s/%d_groundTruth.png' % (result_dir, img_no), X_ori * 255)
        del x_output

    print('\n')
    output_data = "size is %d, Avg PSNR/SSIM for %s is %.2f/%.4f, Epoch number of model is %d \n" % (
    y_size, args.test_data_dir, np.mean(PSNR_All), np.mean(SSIM_All), epoch_num)
    output_file = open(log_file_name, 'a')
    output_file.write(output_data)
    output_file.close()
    print(output_data)

print("CS Reconstruction End")