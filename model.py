
import torch
import torch.nn as nn
import torch.nn.functional as F


# 3d
#########################################################################################################

def conv_block_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Conv3d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        act_fn,
    )
    return model


def conv_trans_block_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.ConvTranspose3d(in_dim,out_dim, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1),output_padding=(0,1,1)),
        #nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1,output_padding=1),
        nn.BatchNorm3d(out_dim),
        act_fn,
    )
    return model


def maxpool_3d():
    pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
    #pool = nn.MaxPool3d(kernel_size=2, stride=2)
    return pool


def conv_block_2_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block_3d(in_dim,out_dim,act_fn),
        nn.Conv3d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.ReLU()
    )
    return model


def conv_block_3_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block_3d(in_dim,out_dim,act_fn),
        conv_block_3d(out_dim,out_dim,act_fn),
        nn.Conv3d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.ReLU()
    )
    return model



class UNet3D_modified(nn.Module):

    def __init__(self, in_dim, out_dim, num_filter):
        super(UNet3D_modified, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter
        act_fn = nn.ReLU()

        print("\n------Initiating U-Net------\n")

        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filter, act_fn)
        self.pool_1 = maxpool_3d()
        self.down_2 = conv_block_2_3d(self.num_filter, self.num_filter * 2, act_fn)
        self.pool_2 = maxpool_3d()

        self.bridge = conv_block_2_3d(self.num_filter * 2, self.num_filter * 8, act_fn)

        self.trans_1 = conv_trans_block_3d(self.num_filter * 8, self.num_filter * 8, act_fn)
        self.up_1 = conv_block_2_3d(self.num_filter * 10, self.num_filter * 4, act_fn)
        self.trans_2 = conv_trans_block_3d(self.num_filter * 4, self.num_filter * 4, act_fn)
        self.up_2 = conv_block_2_3d(self.num_filter * 5, self.num_filter * 1, act_fn)


        self.out = conv_block_3d(self.num_filter, out_dim, act_fn)

    def forward(self, x):

        down_1 = self.down_1(x)
        #print('down_1',down_1.size())
        pool_1 = self.pool_1(down_1)
        #print('pool_1', pool_1.size())
        down_2 = self.down_2(pool_1)
        #print('down_2', down_2.size())
        pool_2 = self.pool_2(down_2)
        #print('pool_2', pool_2.size())


        bridge = self.bridge(pool_2)
        #print('bridge', bridge.size())

        trans_1 = self.trans_1(bridge)
        #print('trans_1',trans_1.size())


        concat_1 = torch.cat([trans_1, down_2], dim=1)
        #print('concat_1', concat_1.size())
        up_1 = self.up_1(concat_1)
        #print('up_1', up_1.size())
        trans_2 = self.trans_2(up_1)
        #print('trans_2', trans_2.size())
        concat_2 = torch.cat([trans_2, down_1], dim=1)
        #print('concat_2', concat_2.size())
        up_2 = self.up_2(concat_2)
        #print('up_2', up_2.size())


        out = self.out(up_2)

        return out



class UNet3D(nn.Module):

    def __init__(self, in_dim, out_dim, num_filter):
        super(UNet3D, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter
        act_fn = nn.ReLU()

        print("\n------Initiating U-Net------\n")

        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filter, act_fn)
        self.pool_1 = maxpool_3d()
        self.down_2 = conv_block_2_3d(self.num_filter, self.num_filter * 2, act_fn)
        self.pool_2 = maxpool_3d()
        self.down_3 = conv_block_2_3d(self.num_filter * 2, self.num_filter * 4, act_fn)
        self.pool_3 = maxpool_3d()

        self.bridge = conv_block_2_3d(self.num_filter * 4, self.num_filter * 8, act_fn)

        self.trans_1 = conv_trans_block_3d(self.num_filter * 8, self.num_filter * 8, act_fn)
        self.up_1 = conv_block_2_3d(self.num_filter * 12, self.num_filter * 4, act_fn)
        self.trans_2 = conv_trans_block_3d(self.num_filter * 4, self.num_filter * 4, act_fn)
        self.up_2 = conv_block_2_3d(self.num_filter * 6, self.num_filter * 2, act_fn)
        self.trans_3 = conv_trans_block_3d(self.num_filter * 2, self.num_filter * 2, act_fn)
        #self.up_3 = conv_block_2_3d(self.num_filter * 3, self.num_filter * 1, act_fn)
        self.up_3 = conv_block_2_3d(self.num_filter * 3, self.num_filter * 1, act_fn)

        self.out = conv_block_3d(self.num_filter, out_dim, act_fn)

    def forward(self, x):
        down_1 = self.down_1(x)
        #print('down_1', down_1.size())
        pool_1 = self.pool_1(down_1)
        #print('pool_1', pool_1.size())
        down_2 = self.down_2(pool_1)
        #print('down_2', down_2.size())
        pool_2 = self.pool_2(down_2)
        #print('pool_2', pool_2.size())

        down_3 = self.down_3(pool_2)
        #print('down_3', down_3.size())
        pool_3 = self.pool_3(down_3)
        #print('pool_3', pool_3.size())

        bridge = self.bridge(pool_3)
        #print('bridge', bridge.size())

        trans_1 = self.trans_1(bridge)
        #print('trans_1', trans_1.size())

        concat_1 = torch.cat([trans_1, down_3], dim=1)
        #print('concat_1', concat_1.size())
        up_1 = self.up_1(concat_1)
        #print('up_1', up_1.size())
        trans_2 = self.trans_2(up_1)
        #print('trans_2', trans_2.size())
        concat_2 = torch.cat([trans_2, down_2], dim=1)
        #print('concat_2', concat_2.size())
        up_2 = self.up_2(concat_2)
        #print('up_2', up_2.size())
        trans_3 = self.trans_3(up_2)
        #print('trans_3', trans_3.size())
        concat_3 = torch.cat([trans_3, down_1], dim=1)
        #print('concat_3', concat_3.size())
        up_3 = self.up_3(concat_3)
        #print('up_3', up_3.size())

        out = self.out(up_3)

        return out


#########################################################################################################


# 2d model

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvInRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ConvInRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=padding, stride=stride)
        self.in2d = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.in2d(x)
        x = self.relu(x)
        return x


class StackEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StackEncoder, self).__init__()
        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3,3),
                                 stride=1, padding=1)
        self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3,3),
                                 stride=1, padding=1)
        self.maxPool = nn.MaxPool2d(kernel_size=(2,2), stride=2)

    def forward(self, x):
        x = self.convr1(x)
        x = self.convr2(x)
        x_trace = x
        x = self.maxPool(x)
        return x, x_trace

class StackDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_size=None):
        super(StackDecoder, self).__init__()

        #self.upSample = nn.Upsample(size=upsample_size, scale_factor=(2,2), mode='bilinear')
        self.upSample = nn.Upsample(scale_factor=(2,2), mode='bilinear')
        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3,3), stride=1, padding=1)
        self.convr2 = ConvBnRelu(in_channels, out_channels, kernel_size=(3,3), stride=1, padding=1)

    def _crop_concat(self, upsampled, bypass):

        margin = bypass.size()[2] - upsampled.size()[2]
        c = margin // 2
        if margin % 2 == 1:
            bypass = F.pad(bypass, (-c,-c-1,-c,-c-1))
        else:
            bypass = F.pad(bypass, (-c,-c,-c,-c))

        return torch.cat((upsampled, bypass), 1)

    def forward(self, x, down_tensor):
        x = self.upSample(x)
        x = self.convr1(x)
        x = self._crop_concat(x, down_tensor)
        x = self.convr2(x)
        return x

class Unet2D(nn.Module):
    def __init__(self, in_shape=(3,512,512)):
        super(Unet2D, self).__init__()
        channels, heights, width = in_shape

        self.down1 = StackEncoder(channels, 64)
        self.down2 = StackEncoder(64, 128)
        self.down3 = StackEncoder(128, 256)
        self.down4 = StackEncoder(256, 512)

        self.center = nn.Sequential(
                        ConvInRelu(512, 1024, kernel_size=(3,3), stride=1, padding=1),
                        ConvInRelu(1024, 1024, kernel_size=(3,3), stride=1, padding=1)
                        )

        #self.up1 = StackDecoder(in_channels=1024, out_channels=512, upsample_size=(56,56))
        #self.up2 = StackDecoder(in_channels=512, out_channels=256, upsample_size=(104,104))
        #self.up3 = StackDecoder(in_channels=256, out_channels=128, upsample_size=(200,200))
        #self.up4 = StackDecoder(in_channels=128, out_channels=64, upsample_size=(392,392))

        self.up1 = StackDecoder(in_channels=1024, out_channels=512)
        self.up2 = StackDecoder(in_channels=512, out_channels=256)
        self.up3 = StackDecoder(in_channels=256, out_channels=128)
        self.up4 = StackDecoder(in_channels=128, out_channels=64)

        self.output_seg_map = nn.Conv2d(64, 1, kernel_size=(1,1), padding=0, stride=1)

    def forward(self, x):



        x, x_trace1 = self.down1(x)
        x, x_trace2 = self.down2(x)
        x, x_trace3 = self.down3(x)
        x, x_trace4 = self.down4(x)

        x = self.center(x)

        x = self.up1(x, x_trace4)
        x = self.up2(x, x_trace3)
        x = self.up3(x, x_trace2)
        x = self.up4(x, x_trace1)

        out = self.output_seg_map(x)
        #out = torch.squeeze(out, dim=1)

        return out
