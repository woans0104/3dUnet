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
import numpy as np
import medpy.metric.binary as med
import argparse
from torchsummary import summary
import dataloader



parser = argparse.ArgumentParser()



parser.add_argument('--input_size',default=256)
parser.add_argument('--batch_size',default=1)
parser.add_argument('--lr',default=0.001)
parser.add_argument('--gpu_id',default='0,1,2,3')
parser.add_argument('--dst_root', default='/data1/archive/lung_seg_location')
parser.add_argument('--exp', type=str)
args = parser.parse_args()




def main():

    # 1.dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
    # make_csv()

    filename = './find_imgdataset1.csv'

    trainset =dataloader.CustomDataset_new(filename)

    #ipdb.set_trace()
    train_loader = data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)



    """
    trainset = datasets.VOCSegmentation("./seg_da/", year='2011', image_set='train',
                                                    download=False, transform=transform, target_transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    testset = datasets.VOCSegmentation("./seg_da/", year='2011', image_set='val',
                                                   download=False, transform=transform, target_transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    """


    # 2. model
    my_net = model.UNet3D(1,1)




    # 3. gpu
    my_net = gpu_select(my_net,args.gpu_id)
    # 4. loss
    criterion = nn.CrossEntropyLoss().cuda()

    # 5. optim
    optimizer = torch.optim.SGD(my_net.parameters(), lr=args.lr, momentum=0.9)

    for epoch in range(90):
            train(train_loader, my_net,criterion, optimizer, epoch)
            #test(test_loader, my_net, criterion, epoch)




def train(train_loader,model,loss_function,optimizer,epoch):
    model.train()
    total_loss=0
    total_jacc=0
    for i, (data, target) in enumerate(train_loader):

        data, target = data.cuda(), target.long().cuda()
        optimizer.zero_grad()
        data,target = Variable(data),Variable(target)


        output = model(data)
        #ipdb.set_trace()
        loss = loss_function(output, target)

        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        acc = jacc(output, target)
        total_jacc+=acc


        if i % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tacc_jaccard: {:.3f}'.format(
                epoch, i * len(data), len(train_loader.dataset),
                100. * i / len(train_loader), loss.item(),acc))

    return total_loss/len(train_loader) , np.round(total_jacc/len(train_loader),3)


def test(test_loader,model,loss_function,epoch,dataset_name=None,visual=None):

    model.eval()

    with torch.no_grad():
        test_loss = 0
        total_acc_jaccard = 0
        total_acc_dice = 0

        for i,(data, target,image_name) in enumerate(test_loader):

            data, target =data.cuda(), target.cuda()
            output = model(data)
            test_loss += loss_function(output, target).item()

            dice_acc =[]
            for j in range(len(output)):
                dice_acc.append(dice(output[j,:,:,:],target[j,:,:,:]))
                #
                print('dice_acc========', dice(output[j,:,:,:],target[j,:,:,:]))
                #print('img_name========', image_name[j])

            dice_acc_ = dice(output, target)
            total_acc_dice += dice_acc_

            jaccard_acc= jacc(output,target)
            total_acc_jaccard +=jaccard_acc

    print("Epoch:%.1f cost : [%.5f] acc_dice: [%.3f]  acc_jaccard: [%.3f]"
          % (epoch, test_loss/len(test_loader), total_acc_dice/len(test_loader), total_acc_jaccard/len(test_loader)))
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
