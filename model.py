
import torch
import torch.nn as nn


def conv_block_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Conv3d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        act_fn,
    )
    return model


def conv_trans_block_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.ConvTranspose3d(in_dim,out_dim, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1),output_padding=(0,1,1)),
        nn.BatchNorm3d(out_dim),
        act_fn,
    )
    return model


def maxpool_3d():
    pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
    return pool


def conv_block_2_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block_3d(in_dim,out_dim,act_fn),
        nn.Conv3d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
    )
    return model


def conv_block_3_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block_3d(in_dim,out_dim,act_fn),
        conv_block_3d(out_dim,out_dim,act_fn),
        nn.Conv3d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
    )
    return model



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
        self.up_3 = conv_block_2_3d(self.num_filter * 3, self.num_filter * 1, act_fn)

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
        down_3 = self.down_3(pool_2)
        #print('down_3', down_3.size())
        pool_3 = self.pool_3(down_3)
        #print('pool_3', pool_3.size())

        bridge = self.bridge(pool_3)
        #print('bridge', bridge.size())

        trans_1 = self.trans_1(bridge)
        #print('trans_1',trans_1.size())


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
