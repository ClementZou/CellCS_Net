from torch.utils.data import Dataset
import torch
import scipy.io as sio
import os
import numpy as np
import math

class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        # torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.xavier_normal_(m.weight.data)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def data_loader_ref(data_dir, y_size, device, phi_label='A'):
    Phi_data_Name = './dataSet/%s/A.mat' % data_dir
    Phi_data = sio.loadmat(Phi_data_Name)
    Phi_input = Phi_data[phi_label][:y_size, :]
    Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor)
    Phi = Phi.to(device)

    X_data_Name = './dataSet/%s/X.mat' % data_dir
    X_data = sio.loadmat(X_data_Name)
    X_labels = X_data['X']

    Y_data_Name = './dataSet/%s/Y.mat' % data_dir
    Y_data = sio.loadmat(Y_data_Name)
    Y_labels = Y_data['Y'][:y_size, :]
    # Y_labels = Y_labels.transpose()

    Qinit_Name = './matrixQ/%s.mat' % data_dir
    if os.path.exists(Qinit_Name):
        Qinit_data = sio.loadmat(Qinit_Name)
        Qinit = Qinit_data['Qinit']
    else:
        X = X_labels.transpose()
        Y_YT = np.dot(Y_labels, Y_labels.transpose())
        X_YT = np.dot(X, Y_labels.transpose())
        Qinit = np.dot(X_YT, np.linalg.inv(Y_YT))
        del X, X_YT, Y_YT
        sio.savemat(Qinit_Name, {'Qinit': Qinit})

    Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor)
    Qinit = Qinit.to(device)

    training_labels = np.concatenate((X_labels, Y_labels.transpose()), axis=1)

    return Phi, training_labels, Qinit


def data_loader_sim(data_dir, y_size, device, phi_label='A'):
    Phi_data_Name = './dataSet/%s/A.mat' % data_dir
    Phi_data = sio.loadmat(Phi_data_Name)
    Phi_input = Phi_data[phi_label][:y_size, :]
    Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor)
    Phi = Phi.to(device)

    X_data_Name = './dataSet/%s/X.mat' % data_dir
    X_data = sio.loadmat(X_data_Name)
    X_labels = X_data['X']

    Qinit_Name = './matrixQ/%s.mat' % data_dir
    if os.path.exists(Qinit_Name):
        Qinit_data = sio.loadmat(Qinit_Name)
        Qinit = Qinit_data['Qinit']
    else:
        X = X_labels.transpose()
        Y = np.dot(Phi_input, X)
        Y_YT = np.dot(Y, Y.transpose())
        X_YT = np.dot(X, Y.transpose())
        Qinit = np.dot(X_YT, np.linalg.inv(Y_YT))
        del X, Y, X_YT, Y_YT
        sio.savemat(Qinit_Name, {'Qinit': Qinit})

    Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor)
    Qinit = Qinit.to(device)

    return Phi, X_labels, Qinit
