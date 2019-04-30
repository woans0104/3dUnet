
import numpy as np
import glob
import os
import pandas as pd
from PIL import Image
import ipdb

def save_log(filename, image, label):
    # fullpath=path+str(filename)

    if type(image) != list and type(image) != np.ndarray:
        image = [image]
        label = [label]

    data = {'img': image, 'lbl': label}
    f = pd.DataFrame(data=data, columns=['img', 'lbl'])
    f.to_csv(filename, index=False)


def find_label_(label):

    # Selects non-255 values ​​from the dataset.


    labels1 = Image.open(label)
    labels2 = np.array(labels1)
    if set(labels2.reshape(-1)) == {0,1}:
        return label



def make_csv_list1():
    #dirname = '/data2/brain_ct/trainvalid_3d'
    dirname = '/data2/brain_ct/test_3d'
    # 경로속 폴더
    filenames = os.listdir(dirname)

    image = []
    label = []
    for filename in filenames:

        # 폴더 1개의 모든 파일을 읽고
        full_filename = os.path.join(dirname, filename)
        image_folder = sorted(glob.glob(full_filename+'/*'))

        image_list = []
        label_list = []

        # 폴더 1개의 모든 파일에서
        for i in image_folder:
            ext = os.path.splitext(i)[-1]
            # 라벨이면
            if ext == '.gif':
                # 255 값 찾기
                f_lb=find_label_(i)
                if f_lb != None:
                    label_list.append(f_lb)
                    f_img = os.path.splitext(i)[0].split('_mask')[0] + '.png'
                    if f_img in image_folder:
                        image_list.append(f_img)

        if len(image_list)%3 !=0:
            delete = len(image_list)%3
            for h in range(delete):
                del image_list[-1]
                del label_list[-1]

        image.append(image_list)
        label.append(label_list)

    filename = './find_testdataset3.csv'
    save_log(filename,image,label)

    return image,label


def make_listindex(numv,Flist):
    # 폴더 1개의 모든 파일을 인자로 받음
    make_lmgindex = []
    make_labelindex = []
    count = 0 #255 값을 찾은 라벨의 index
    numv = numv # 주위의 몇개의 이미지를 사용할 것인가에 대한 변수
    thres = numv // 2
    while count + numv <= len(Flist):
        # 라벨이면
        if os.path.splitext(Flist[count])[-1] == '.gif':
            # 255 값 찾기
            f_lb = find_label_(Flist[count])
            if f_lb != None:
                count =  Flist.index(f_lb)
                # index가 맨 앞이면 앞의 이미지를 못가져오므로
                if count - thres >= 0:
                    holsu1 = []
                    holsu2 = []
                    for j in range(numv):
                        holsu1.append(Flist[count - thres + j])
                        # image path가 _mask만 빼면 똑같기에
                        holsu2.append(Flist[count - thres + j].split('_mask')[0] + '.png')

                    make_lmgindex.append(holsu2)
                    make_labelindex.append(holsu1)
                    count += numv
                else:
                    count += 1
            else:
                count += 1

    return make_lmgindex, make_labelindex


################################################################################

def make_csv_list():
    #dirname = '/data2/brain_ct/trainvalid_3d'
    dirname = '/data2/brain_ct/trainvalid_3d'
    # 경로속 폴더
    filenames = os.listdir(dirname)

    image = []
    label = []
    for filename in filenames:

        # 폴더 1개의 모든 파일을 읽고
        full_filename = os.path.join(dirname, filename)
        image_folder = sorted(glob.glob(full_filename+'/*.png'))
        target_folder = sorted(glob.glob(full_filename + '/*.gif'))

        imglist,labellist = make_listindex(3,target_folder)

        image.append(imglist)
        label.append(labellist)


    filename = './find_imgdataset3.csv'
    save_log(filename,image,label)

    return image,label

################################################################################

if __name__ == '__main__':
    make_csv_list()