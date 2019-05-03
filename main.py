import argparse
import torch.nn as nn
import torch
from torch.optim import lr_scheduler
import torch.utils.data as data
import ipdb
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import model
import utill
import numpy as np
import medpy.metric.binary as med
import argparse
from torchsummary import summary
import dataloader
import matplotlib.pyplot as plt
from torch.autograd import Function
from itertools import repeat
from medpy.metric import binary

parser = argparse.ArgumentParser()

parser.add_argument('--data_mode',default='2d',help='2d,3d')
parser.add_argument('--batch_size',default=4)
parser.add_argument('--epoch',default=50)
parser.add_argument('--lr',default=0.01)
parser.add_argument('--gpu_id',default='0,1')
parser.add_argument('--dst_root', default='/data1/archive/lung_seg_location')

parser.add_argument('--train_mode',default=False,type=str)
parser.add_argument('--test_mode',default=False,type=str)

args = parser.parse_args()



def main():

    # 1.dataset

    train_filename = './data_csv/train_dataset.csv'
    val_filename = './data_csv/test_dataset.csv'


    if args.data_mode == '2d':
        trainset = dataloader.CustomDataset2d(train_filename,mode='Train')
        train_loader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        valset = dataloader.CustomDataset2d(val_filename,mode='val')
        val_loader = data.DataLoader(valset, batch_size=1, shuffle=True, num_workers=4)

        # testset = dataloader.CustomDataset(test_filename)
        # test_loader = data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=4)

    elif args.data_mode == '3d':
        trainset = dataloader.CustomDataset3d(train_filename)
        train_loader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        valset = dataloader.CustomDataset3d(val_filename)
        val_loader = data.DataLoader(valset, batch_size=1, shuffle=True, num_workers=4)

        # testset = dataloader.CustomDataset(test_filename)
        # test_loader = data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=4)



    # 2. model

    if args.data_mode == '2d':
        my_net= model.Unet2D()
    elif  args.data_mode == '3d':
        my_net = model.UNet3D_modified(in_dim=1,out_dim=1,num_filter=4)



    # 3. gpu
    my_net = gpu_select(my_net,args.gpu_id)
    # 4. loss
    criterion = nn.BCEWithLogitsLoss().cuda()

    # 5. optim
    optimizer = torch.optim.SGD(my_net.parameters(), lr=args.lr, momentum=0.9)
    #optimizer = torch.optim.Adam(my_net.parameters(), lr=args.lr)


    """
    for epoch in range(args.epoch):
       # train_loss, train_acc= train(train_loader, my_net,criterion, optimizer, epoch)
        val_loss, val_acc = test(val_loader, my_net, criterion, epoch)


        train_lossfilename = './log_csv/train_loss.csv'
        train_accfilename = './log_csv/train_acc.csv'
        val_lossfilename = './log_csv/val_loss.csv'
        val_accfilename = './log_csv/val_acc.csv'

        #utill.save_MultiLog(train_lossfilename, epoch,train_loss)
        #utill.save_MultiLog(train_accfilename, epoch, train_acc)
        #utill.save_MultiLog(val_lossfilename, epoch, val_loss,)
        #utill.save_MultiLog(val_accfilename, epoch, val_acc)

    model_name ='2dunet'
    torch.save(my_net.state_dict(), './model_{}_{}_{}.pth'.format(model_name, args.lr, args.epoch))

    """

   # my_net = model.Unet2D().cuda()
    my_net.load_state_dict(torch.load('./model_2dunet_0.01_50.pth'))
    test(val_loader, my_net, criterion,1)


    # test
    if args.test_mode == 'True':
        print('test go')
        # load parameter
        my_net.load_state_dict(torch.load('./model_2dunet_0.01_50.pth'))
        test(val_loader,my_net,criterion)






def train(train_loader,model,loss_function,optimizer,epoch):
    model.train()
    total_loss=0
    total_jacc=0
    for i, (data, target,_) in enumerate(train_loader):

        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        data,target = Variable(data),Variable(target)


        output = model(data)
        loss = loss_function(output, target)

        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        jac, dice_package = perpomance(output, target)
        total_jacc+=jac


        if i % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tacc_jaccard: {:.3f}'.format(
                epoch, i * len(data), len(train_loader.dataset),
                100. * i / len(train_loader), loss.item(),jac))

    return total_loss/len(train_loader) , np.round(total_jacc/len(train_loader),3)


def test(test_loader,model,loss_function,epoch,dataset_name=None,visual=None):

    model.eval()

    with torch.no_grad():
        test_loss = 0
        total_acc_jaccard = 0
        total_acc_dice = 0

        for i,(data, target,target_name) in enumerate(test_loader):

            print('target_name',target_name)
            data, target =data.cuda(), target.cuda()
            output = model(data)
            test_loss += loss_function(output, target).item()

            """
            if args.data_mode =='2d':
                dice_acc =[]
                for j in range(len(output)):
                    jac, dice_package = perpomance(output[j, :, :, :], target[j, :, :, :])
                    dice_acc.append(jac)

                    print('''''''''''''''''''''''')
                    print('jac', jac)
                    print('dice_package', dice_package)
                    print('''''''''''''''''''''''')

                # visualize

                for h in range(len(dice_acc)):
                    utill.visualize_img([dice_acc[h], target_name[h], output[h, :, :, :]], model)
            """


            if args.data_mode =='3d':
                dice_acc = []
                for j in range(len(output[0,0,:,:])):
                    jac, dice_package = perpomance(output[0,0, j, :], target[0,0, j, :])
                    dice_acc.append(jac)
                    ipdb.set_trace()
                    print('''''''''''''''''''''''')
                    print('jac', jac)
                    print('dice_package', dice_package)
                    print('''''''''''''''''''''''')

                # visualize
                for h in range(len(dice_acc)):

                    utill.visualize_img([dice_acc[h], target_name[h], output[0, 0, h, :]], model)



            jac, dice_package = perpomance(output, target)
            total_acc_dice += dice_package
            total_acc_jaccard +=jac

    print("Epoch:%.1f cost : [%.5f] acc_dice: [%.3f]  acc_jaccard: [%.3f]"
          % (epoch, test_loss/len(test_loader), total_acc_dice/len(test_loader), total_acc_jaccard/len(test_loader)))
    ipdb.set_trace()
    return test_loss/len(test_loader),np.round(total_acc_jaccard/len(test_loader),3)



def gpu_select(my_net, gpu_id):

    # 4.gpu select
    if len(gpu_id) != 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        gpu_list = []
        count = 0
        for i in gpu_id.split(','):
            gpu_list.append(count)
            count += 1
        print(gpu_list)
        my_net = nn.DataParallel(my_net.cuda(), device_ids=gpu_list)

        return my_net
    else:

        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        my_net.cuda()

        return my_net




######################################################################################################


def perpomance(input, target):

    input_re = F.sigmoid(input)
    input_re = input_re.cpu().data.numpy().squeeze()
    target_re = target.cpu().data.numpy().squeeze()

    thresh_value = 0.5
    thresh_image = input_re > thresh_value

    intersection = (input_re * target_re).sum()
    union = (input_re.sum() + target_re.sum()) - intersection

    dice =  (2 * intersection) / (input_re.sum() + target_re.sum())
    jacc = intersection / union

    #dice_package = binary.dc(input_re, target_re)
    # assd = med.assd(input1,target1)



    return jacc,dice





def dice(input, taget):

    # re range

    input_re = F.sigmoid(input)
    input_re = input_re.cpu().data.numpy()

    #print(np.min(input_re))
    #print(np.max(input_re))

    thresh_value = 0.5
    thresh_image = input_re > thresh_value

    thresh_targetValue = 0.5
    thresh_targetImage = taget.cpu().data.numpy() > thresh_targetValue

    input1 = thresh_image.reshape(-1)
    target1 = thresh_targetImage.reshape(-1)

    return 2 * (input1 * target1).sum() / (input1.sum() + target1.sum())


def jacc(input, taget):

    # re range
    input_re = F.sigmoid(input)
    input_re = input_re.cpu().data.numpy()


    thresh_value = 0.5
    thresh_image = input_re > thresh_value

    thresh_targetValue = 0.5
    thresh_targetImage = taget.cpu().data.numpy() > thresh_targetValue


    input1 = thresh_image.reshape(-1)
    target1 = thresh_targetImage.reshape(-1)

    intersection=(input1 * target1).sum()
    union=(input1.sum() + target1.sum())-intersection

    return  (intersection / union)


if __name__ == '__main__':
    main()
