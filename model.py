from numpy import pad
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# Define ISTA-Net-plus Block
class BasicBlock(torch.nn.Module):
    def __init__(self, Nf=128, stage=2):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))
        self.lambda_res = nn.Parameter(torch.Tensor([0.3]))
        if stage == 1:
            self.lambda_res = nn.Parameter(torch.Tensor([0]))
        self.stage = stage

        self.conv_d = nn.Conv2d(1, Nf, 3, padding=1, bias=True)
        self.conv1 = nn.Conv2d(Nf, Nf, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(Nf, Nf, 3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(Nf, Nf, 3, padding=1, bias=True)
        self.conv4 = nn.Conv2d(Nf, Nf, 3, padding=1, bias=True)
        self.conv_g = nn.Conv2d(Nf, 1, 3, padding=1, bias=True)

        self.conv_fusion1 = nn.Conv2d(Nf, Nf, 3, padding=1, bias=True)
        self.conv_fusion2 = nn.Conv2d(Nf, Nf, 3, padding=1, bias=True)
        self.conv_fusion3 = nn.Conv2d(Nf, Nf, 3, padding=1, bias=True)
        self.conv_fusion4 = nn.Conv2d(Nf * 2, Nf, 1, padding=0, bias=True)

    def forward(self, x, PhiTPhi, PhiTb, x_forward_, x_backward_):
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x_input = x.view(-1, 1, 128, 128)

        x_D = self.conv_d(x_input)
        x_D = self.fusion(x_D, x_backward_)
        x = F.relu(self.conv1(x_D))
        x_forward = self.conv2(x)  # self.conv2(x) * (1 - self.lambda_res) + x_forward_ * self.lambda_res

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.relu(self.conv3(x))
        x_backward = self.conv4(
            x)  # self.conv4(F.relu(self.conv3(x)))# * (1 - self.lambda_res) + self.lambda_res * x_backward_
        x_G = self.conv_g(x_backward)
        # print(self.stage)
        # print('x_forward backward G: ', x_forward.shape, x_backward.shape, x_G.shape)
        x_pred = x_input + x_G
        x_pred = x_pred.view(-1, 16384)

        x_D_est = self.conv4(F.relu(self.conv3(x_forward)))
        symloss = x_D_est - x_D

        return [x_pred, symloss, x_forward, x_backward]

    def fusion(self, x, x_pre):
        x_fusion = F.softmax(self.conv_fusion2(self.conv_fusion1(x)), dim=1)
        x_pre_fusion = self.conv_fusion3(x_pre)
        # print(self.stage)
        # print('x_fusion pre_fusion pre: ', x_fusion.shape, x_pre_fusion.shape, x_pre.shape)
        x_fusion = torch.mul(x_fusion, x_pre_fusion)
        x_fusion = torch.concat((x, x_fusion), dim=1)
        x_fusion = self.conv_fusion4(x_fusion)
        return x_fusion


# Define ISTA-Net-plus
class ISTANetplus(torch.nn.Module):
    def __init__(self, LayerNo, Nf=128):
        super(ISTANetplus, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.zero_tensor = torch.Tensor([0])
        self.Nf = Nf

        onelayer.append(BasicBlock(Nf, 1))
        for i in range(LayerNo):
            onelayer.append(BasicBlock(Nf, i + 2))

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit):

        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

        layers_sym = []  # for computing symmetric loss

        [x, layer_sym, x_forward, x_backward] = self.fcs[0](x, PhiTPhi, PhiTb,
                                                            torch.zeros(x.shape[0], self.Nf, 128, 128).cuda(),
                                                            torch.zeros(x.shape[0], self.Nf, 128, 128).cuda())
        layers_sym.append(layer_sym)

        for i in range(self.LayerNo - 1):
            [x, layer_sym, x_forward, x_backward] = self.fcs[i + 1](x, PhiTPhi, PhiTb, x_forward, x_backward)
            layers_sym.append(layer_sym)

        x_final = x

        return [x_final, layers_sym]