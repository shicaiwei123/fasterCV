import torch
import torch.nn as nn
import math


class ChannelAttention(nn.Module):
    '''
    通道注意力模块
    '''

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    '''
    空间注意力模块
    '''

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM_Module(nn.Module):

    def __init__(self, channels, reduction):
        super(CBAM_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)

        self.fc3 = nn.Linear(8, 4, bias=True)
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_after_concat = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention module:（Mc(f) = σ(MLP(AvgPool(f)) + MLP(MaxPool(f)))）
        module_input = x
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x = avg + mx

        x = self.sigmoid_channel(x)
        # Spatial attention module:Ms (f) = σ( f7×7( AvgPool(f) ; MaxPool(F)] )))
        x = module_input * x
        module_input = x
        avg = torch.mean(x, 1, keepdim=True)
        mx, _ = torch.max(x, 1, keepdim=True)
        x = torch.cat((avg, mx), 1)
        x = self.conv_after_concat(x)
        x = self.sigmoid_spatial(x)
        x = module_input * x
        return x


class Flatten(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dim=dim

    def forward(self, x):
        x = torch.flatten(x, self.dim)
        return x

class Discriminator(nn.Module):
    def __init__(self, args, input_dims):
        super(Discriminator, self).__init__()
        self.args = args
        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(input_dims, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2),
            # nn.LogSoftmax()
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out


class SPP(nn.Module):
    def __init__(self, merge='max'):
        super(SPP, self).__init__()
        if merge == 'max':
            self.normal_pooling = nn.AdaptiveMaxPool2d((4, 4))
            self.pooling_2x2 = nn.AdaptiveMaxPool2d((2, 2))
            self.pooling_1x1 = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.normal_pooling = nn.AdaptiveAvgPool2d((4, 4))
            self.pooling_2x2 = nn.AdaptiveAvgPool2d((2, 2))
            self.pooling_1x1 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x_normal = self.normal_pooling(x)
        x_2x2 = self.pooling_2x2(x_normal)
        x_1x1 = self.pooling_1x1(x_normal)

        x_normal_flatten = torch.flatten(x_normal, start_dim=2, end_dim=3)  # B X C X feature_num

        x_2x2_flatten = torch.flatten(x_2x2, start_dim=2, end_dim=3)

        x_1x1_flatten = torch.flatten(x_1x1, start_dim=2, end_dim=3)

        x_feature = torch.cat((x_normal_flatten, x_2x2_flatten, x_1x1_flatten), dim=2)

        return x_feature


def init_weights(m):
    print(m)
    if type(m) == nn.Linear:
        print(m.weight)
    else:
        print('error')


class MMTM(nn.Module):
    def __init__(self, dim_visual, dim_skeleton, ratio):
        super(MMTM, self).__init__()
        dim = dim_visual + dim_skeleton
        dim_out = int(2 * dim / ratio)
        self.fc_squeeze = nn.Linear(dim, dim_out)

        self.fc_visual = nn.Linear(dim_out, dim_visual)
        self.fc_skeleton = nn.Linear(dim_out, dim_skeleton)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # initialize
        with torch.no_grad():
            self.fc_squeeze.apply(init_weights)
            self.fc_visual.apply(init_weights)
            self.fc_skeleton.apply(init_weights)

    def forward(self, visual, skeleton):
        squeeze_array = []
        for tensor in [visual, skeleton]:
            # print(tensor.shape)
            tview = tensor.view(tensor.shape[:2] + (-1,))
            squeeze_array.append(torch.mean(tview, dim=-1))
        squeeze = torch.cat(squeeze_array, 1)

        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)

        vis_out = self.fc_visual(excitation)
        sk_out = self.fc_skeleton(excitation)

        vis_out = self.sigmoid(vis_out)
        sk_out = self.sigmoid(sk_out)

        dim_diff = len(visual.shape) - len(vis_out.shape)
        vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)

        dim_diff = len(skeleton.shape) - len(sk_out.shape)
        sk_out = sk_out.view(sk_out.shape + (1,) * dim_diff)

        return visual * vis_out * skeleton * sk_out
