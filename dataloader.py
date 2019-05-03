import torch.utils.data as data
import numpy as np
import ipdb
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import ast


class CustomDataset3d(Dataset):

    def __init__(self, csv_file):

        self.cxrs_frame = pd.read_csv(csv_file, delimiter=',')
        self.transforms1 = transforms.ToTensor()
        self.transforms2 = transforms.Normalize([0.5],[0.5])
        self.image ,self.target = self.make_tupleset(self.cxrs_frame)
        self.count = self.check_file(sorted(self.image),sorted(self.target))



    def make_tupleset(self,dataframe):
        """
            It is a task to make it into a form to be called by get_item of the data loader.

            return imglist,labellist

            imglist[0] =
            [/data2/brain_ct/trainvalid_3d/435/FILE0010.png,
            /data2/brain_ct/trainvalid_3d/435/FILE0011.png,
            /data2/brain_ct/trainvalid_3d/435/FILE0012.png]
        """

        # Since the value is str, replace it with list.
        for i in range(len(dataframe)):
            dataframe.iloc[i][0] = ast.literal_eval(dataframe.iloc[i][0])
            dataframe.iloc[i][1] = ast.literal_eval(dataframe.iloc[i][1])

        aa = dataframe.values

        imglist = []
        labellist = []
        for i in range(len(aa)):
            for j in range(len(aa[i])):
                for k in range(len(aa[i][j])):
                    imglist.append(aa[i][0][k])
                    labellist.append(aa[i][1][k])

        return imglist,labellist


    def make_img(self,image):
        # image processing

        image_ = Image.open(image).convert('L')
        image_ = np.array(image_, dtype=np.uint8)
        image_ = self.transforms1(image_)
        image_ = self.transforms2(image_)
        image_ = image_.unsqueeze(0)

        return image_

    def make_label(self, label):
        # label processing

        mask_ = Image.open(label).convert('L')
        mask_ = np.array(mask_, dtype=np.uint8)

        # When transforming to torch, it divides 255, so first multiply by 255.
        if np.max(mask_.reshape(-1)) == 1:
            mask_ = mask_ * 255

        mask_ = self.transforms1(mask_)
        mask_ = mask_.unsqueeze(0)

        return mask_


    def check_file(self,imagename,targetname):

        count = 0
        for i in range(len(imagename)):
            img = np.array(imagename[i][0].split('/')[-2] + imagename[i][0].split('/')[-1].split('.')[0])
            tar =  np.array(targetname[i][0].split('/')[-2] + targetname[i][0].split('/')[-1].split('_mask')[0])
            if img != tar:
                count+=1
                print('image :: ',img)
                print('target :: ' , tar)
        print('count==',count)
        return count

    def __getitem__(self, index):

        image = sorted(self.image)
        target = sorted(self.target)


        image_frame = image[index][0]
        image_frame1 = image[index][1]
        image_frame2 = image[index][2]

        target_frame = target[index][0]
        target_frame1 = target[index][1]
        target_frame2 = target[index][2]

        image_ = self.make_img(image_frame)
        image_1 = self.make_img(image_frame1)
        image_2 = self.make_img(image_frame2)

        mask_ = self.make_label(target_frame)
        mask_1 = self.make_label(target_frame1)
        mask_2 = self.make_label(target_frame2)


        # make 3d image
        image_3d = torch.cat([image_,image_1, image_2], 1)
        mask_3d = torch.cat([mask_, mask_1, mask_2], 1)

        return image_3d,mask_3d,target_frame1

    def __len__(self):  # return count of sample we have
        return len(self.target)




class CustomDataset2d(Dataset):

    def __init__(self, csv_file,mode):

        self.cxrs_frame = pd.read_csv(csv_file, delimiter=',')
        self.transforms1 = transforms.ToTensor()
        self.transforms2 = transforms.Normalize([0.5],[0.5])
        self.image ,self.target = self.make_tupleset(self.cxrs_frame,mode)
        self.count = self.check_file(sorted(self.image),sorted(self.target))


    def make_tupleset(self,dataframe,mode = 'Train'):
        """
            It is a task to make it into a form to be called by get_item of the data loader.

            return imglist,labellist

            imglist[0] =
            [/data2/brain_ct/trainvalid_3d/435/FILE0010.png,
            /data2/brain_ct/trainvalid_3d/435/FILE0011.png,
            /data2/brain_ct/trainvalid_3d/435/FILE0012.png]
        """

        # Since the value is str, replace it with list.
        for i in range(len(dataframe)):
            dataframe.iloc[i][0] = ast.literal_eval(dataframe.iloc[i][0])
            dataframe.iloc[i][1] = ast.literal_eval(dataframe.iloc[i][1])

        aa = dataframe.values


        if mode == 'val':
            imglist = []
            labellist = []
            for i in range(len(aa)):
                imglist.append(aa[i][0][0])
                labellist.append(aa[i][1][0])

        if mode =='Train':
            imglist = []
            labellist = []
            for i in range(len(aa)):
                for j in range(len(aa[i])):
                    for k in range(len(aa[i][j])):
                        imglist.append(aa[i][0][k])
                        labellist.append(aa[i][1][k])

        return imglist,labellist


    def make_img(self,image):
        # image processing

        image_ = Image.open(image).convert('L')
        image_ = np.array(image_, dtype=np.uint8)
        image_ = self.transforms1(image_)
        image_ = self.transforms2(image_)
        #image_ = image_.unsqueeze(0)

        return image_

    def make_label(self, label):
        # label processing

        mask_ = Image.open(label).convert('L')
        mask_ = np.array(mask_, dtype=np.uint8)

        # When transforming to torch, it divides 255, so first multiply by 255.
        if np.max(mask_.reshape(-1)) == 1:
            mask_ = mask_ * 255

        mask_ = self.transforms1(mask_)
        #mask_ = mask_.unsqueeze(0)

        return mask_


    def check_file(self,imagename,targetname):

        count = 0
        for i in range(len(imagename)):
            img = np.array(imagename[i][0].split('/')[-2] + imagename[i][0].split('/')[-1].split('.')[0])
            tar =  np.array(targetname[i][0].split('/')[-2] + targetname[i][0].split('/')[-1].split('_mask')[0])
            if img != tar:
                count+=1
                print('image :: ',img)
                print('target :: ' , tar)
        print('count==',count)
        return count

    def __getitem__(self, index):

        image = sorted(self.image)
        target = sorted(self.target)

        image_frame = image[index][0]
        image_frame1 = image[index][1]
        image_frame2 = image[index][2]

        #target_frame = target[index][0]
        target_frame1 = target[index][1]
        #target_frame2 = target[index][2]

        image_ = self.make_img(image_frame)
        image_1 = self.make_img(image_frame1)
        image_2 = self.make_img(image_frame2)

        #mask_ = self.make_label(target_frame)
        mask_1 = self.make_label(target_frame1)
        #mask_2 = self.make_label(target_frame2)


        # make 3d image
        image_3d = torch.cat([image_,image_1, image_2], 0)
        #mask_3d = torch.cat([mask_, mask_1, mask_2], 0)

        return image_3d,mask_1,target_frame1

    def __len__(self):  # return count of sample we have
        return len(self.target)




