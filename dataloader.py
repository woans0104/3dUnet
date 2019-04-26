import torch.utils.data as data
import numpy as np
import ipdb
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset




class CustomDataset_old(Dataset):

    def __init__(self, csv_file):  # initial logic happens like transform

        self.cxrs_frame = pd.read_csv(csv_file, delimiter=',')
        self.transforms1 = transforms.ToTensor()
        self.transforms2 = transforms.Normalize([0.5],[0.5])


    def __getitem__(self, index):

        # indexing test

        image_frame = self.cxrs_frame.iloc[index][0].split(',')
        target_frame = self.cxrs_frame.iloc[index][1].split(',')

        image_name = []
        target_name = []
        for i in range(len(image_frame)):
            image_name.append(image_frame[i].strip("[").strip(' ').strip("]").strip("'"))
            target_name.append(target_frame[i].strip("[").strip(' ').strip("]").strip("'"))


        divider = 3
        start_pos = 0
        image3d_name = []
        target3d_name = []

        for i in range(start_pos,len(image_name),divider):
            image = image_name[start_pos:start_pos+divider]
            mask = target_name[start_pos:start_pos+divider]
            start_pos = start_pos + divider

            image_test =[]
            mask_test = []
            for j in range(len(image)):

                mask_= np.array(Image.open(mask[j]).convert('L'),dtype=np.uint8)

                if np.max(mask_.reshape(-1)) == 1:
                   mask_ = mask_*255


                mask_ = self.transforms1(mask_)
                image_ =  self.transforms2(self.transforms1(np.array(Image.open(image[j]).convert('L'),dtype=np.uint8)))

                mask_ = mask_.unsqueeze(0)
                image_ = image_.unsqueeze(0)

                image_test.append(image_)
                mask_test.append(mask_)


            image_3d = torch.cat([image_test[0],image_test[1],image_test[2]],0)
            mask_3d = torch.cat([mask_test[0], mask_test[1], mask_test[2]], 0)

            image3d_name.append(image_3d)
            target3d_name.append(mask_3d)

        return image3d_name, target3d_name

    def __len__(self):  # return count of sample we have

        return len(self.cxrs_frame)



class CustomDataset_new(Dataset):

    def __init__(self, csv_file):  # initial logic happens like transform

        self.cxrs_frame = pd.read_csv(csv_file, delimiter=',')
        self.transforms1 = transforms.ToTensor()
        self.transforms2 = transforms.Normalize([0.5],[0.5])
        self.datalist = []
        self.count = 0

    def __getitem__(self, index):

        image_frame = self.cxrs_frame.iloc[index][0]
        target_frame = self.cxrs_frame.iloc[index][1]

        mask_= Image.open(target_frame).convert('L')
        image_ = Image.open(image_frame).convert('L')


        #resize
        resize = transforms.Resize(size=(128, 128))
        image_ = resize(image_)
        mask_ = resize(mask_)


        image_ = np.array(image_, dtype=np.uint8)
        mask_ = np.array(mask_, dtype=np.uint8)
        if np.max(mask_.reshape(-1)) == 1:
            mask_ = mask_ * 255


        image_= self.transforms1(image_)
        image_ = self.transforms2(image_)
        mask_ = self.transforms1(mask_)

        mask_ = mask_.unsqueeze(0)
        image_ = image_.unsqueeze(0)

        #image_3d = torch.cat([image3d_name[0],image3d_name[1],image3d_name[2]],1)
        #mask_3d = torch.cat([target3d_name[0], target3d_name[1], target3d_name[2]], 1)

        return mask_, image_,image_frame

    def __len__(self):  # return count of sample we have
        return len(self.cxrs_frame)



class MyDataset(Dataset):
    def __init__(self, image_paths, target_paths, train=True):
        self.image_paths = image_paths
        self.target_paths = target_paths


    def __getitem__(self, index):
        image = self.image_paths[index]
        mask = self.target_paths[index]

        return image, mask

    def __len__(self):
        return len(self.image_paths)




if __name__ == '__main__':

    """
    filename = './find_testdataset.csv'
    id = CustomDataset_new(filename)

    image3d_name=[]
    target3d_name=[]
    for i in range(len(id)):
        target3d_name.append(id[0][0])
        image3d_name.append(id[0][1])


    divider = 3
    start_pos = 0
    image3d_name1 = []
    target3d_name1 = []
    for i in range(start_pos, len(image3d_name), divider):
        image = image3d_name[start_pos:start_pos + divider]
        mask = target3d_name[start_pos:start_pos + divider]
        start_pos = start_pos + divider

        image_3d = torch.cat([image[0], image[1], image[2]], 1)
        mask_3d = torch.cat([mask[0], mask[1], mask[2]], 1)

        image3d_name1.append(image_3d)
        target3d_name1.append(mask_3d)



    dataset3d = MyDataset(image3d_name1,target3d_name1)
    train_loader = data.DataLoader(dataset3d, batch_size=1, shuffle=True, num_workers=4)
    ipdb.set_trace()

"""

