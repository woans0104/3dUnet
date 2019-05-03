import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import matplotlib.patches as mpatches
import glob
import torch.nn.functional as F
import ipdb



def make_csvfile(filename, image, label):
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

def make_csv_list(val_mode =False):
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


    if val_mode:
        train_filename = './train_dataset4.csv'
        val_filename = './val_dataset4.csv'
        val_size = int(len(image)*0.2)

        make_csvfile(train_filename, image[val_size:], label[val_size:])
        make_csvfile(val_filename, image[:val_size], label[:val_size])
    else:
        filename = './train_dataset.csv'
        make_csvfile(filename, image, label)




def make_testCsv_list():
    #dirname = '/data2/brain_ct/trainvalid_3d'
    dirname = '/data2/brain_ct/test_3d'
    # 경로속 폴더
    filenames = os.listdir(dirname)

    image = []
    label = []
    for filename in filenames:

        # 폴더 1개의 모든 파일을 읽고
        full_filename = os.path.join(dirname, filename)
        image_folder = sorted(glob.glob(full_filename+'/*.png'))
        target_folder = sorted(glob.glob(full_filename + '/*.gif'))

        divider = 3
        start_pos = 0
        for i in range(start_pos, len(image_folder), divider):
            image_ = image_folder[start_pos:start_pos + divider]
            label_ = target_folder[start_pos:start_pos + divider]
            if len(image_) == divider:
                image.append([image_])
                label.append([label_])
            start_pos = start_pos + divider

    filename = './test_dataset.csv'
    make_csvfile(filename, image, label)


################################################################################

# log file save
def save_MultiLog(filename, epoch,loss,acc=False ):
    if acc:
        if os.path.exists(filename):
            f = pd.read_csv(filename)

            if type(loss) is list or type(loss) is np.ndarray:
                print('check')
                for i in range(len(loss)):
                    f = f.append([{'epoch': epoch[i], 'loss': loss[i],'acc': acc[i]}], ignore_index=True)
            else:
                f = f.append([{'epoch': epoch, 'loss': loss,'acc': acc}], ignore_index=True)
            f.to_csv(filename, index=False)
        else:
            if type(loss) != list and type(loss) != np.ndarray:
                loss = [loss]
                acc = [acc]
                epoch = [epoch]
            data = {'epoch': epoch, 'loss': loss,'acc': acc}
            f = pd.DataFrame(data=data, columns=['epoch', 'loss','acc'])
            f.to_csv(filename, index=False)
    else:
        save_SingleLog(filename,epoch,loss)

def save_SingleLog(filename,epoch, loss ):
    col_name = filename.split('_')[-1].split('.')[0]
    if os.path.exists(filename):
        f = pd.read_csv(filename)
        #print(f)
        if type(loss) is list or type(loss) is np.ndarray:
            print('check')
            for i in range(len(loss)):
                f = f.append([{'epoch': epoch[i], '{}'.format(col_name): loss[i]}], ignore_index=True)
        else:
            f = f.append([{'epoch': epoch, '{}'.format(col_name): loss}], ignore_index=True)
        f.to_csv(filename, index=False)

    else:
        if type(loss) != list and type(loss) != np.ndarray:
            loss = [loss]
            epoch = [epoch]
        data = {'epoch': epoch, '{}'.format(col_name): loss}
        f = pd.DataFrame(data=data, columns=['epoch', '{}'.format(col_name)])
        f.to_csv(filename, index=False)




######################################################################################################
"""

visuallize img

"""


def visualize_img(img_list,model):
    """
    This function returns a segmentation image for the model's output.

    1. img_list : [dice_acc,image_name,output]
    2. model : need output img name tag
    3. dataset_name : img name
    4. mode : need output img name tag & dice max or min img

    """
    print('this is img_list')
    #print(img_list)
    #print(img_list[0])
    #print(img_list[1])

    img = F.sigmoid(img_list[2])

    img = img.permute(1,2,0) # permute : change dimensions
    #print('permute:::::::',img.shape)
    img = img.cpu().data.numpy().reshape(512,512)

    thresh_value = 0.5
    thresh_image = img > thresh_value

    model_name = str(model).split('(')[0]


    aa = np.round(img_list[0], 3)
    print('==========================================')
    print('np.round(img_list[0], 3)',aa)
    print('==========================================')

    try:
        Directory_name='./output_img/{}/'.format( model_name)
        if not (os.path.isdir(Directory_name)):
            os.makedirs(os.path.join(Directory_name))

        plt.imsave(
            Directory_name + '{}_{}.png'.format(str(aa),img_list[1].split('/')[-2]+'_'+img_list[1].split('/')[-1].split('.')[0]),
                                                thresh_image, cmap='gray')
    except OSError as e:
        if e.errno != e.errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise #make error


    print('visual end')


def make_contours(image):
    """
    This function returns the outline of the image.

    """
    # if channel is not 1
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    except:

        print('ch =1')

    # Using Thresholds to Binary Images
    #print(set(image.reshape(-1)))

    ret, thr = cv2.threshold(image, 0.5, 255, cv2.THRESH_BINARY)


    # image, contours ,
    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def make_visualize(ori_img,target_img,pred_img,path):
    """
    ori : use visualize
    pred : use contours => fill image(blue)
    target : use contours => contours image(red)

    """

    print(ori_img.split('/')[-1].split('.')[0].split('_')[1])
    print(pred_img.split('/')[-1].split('_')[2])
    print(target_img.split('/')[-1].split('.')[0].split('_')[1])




    if ori_img.split('/')[-1].split('.')[0].split('_')[1] != pred_img.split('/')[-1].split('_')[2] and target_img.split('/')[-1].split('.')[0].split('_')[1] != pred_img.split('/')[-1].split('_')[2] :
        print("+++++ not same")



    fig_name = pred_img.split('/')[-1]
    visual_name = pred_img.split('/')[-1].split('.png')[0]

    ori_img = Image.open(ori_img).convert('RGB')
    target_img = Image.open(target_img).convert("L")
    pred_img = Image.open(pred_img).convert("L")


    #resize
    ori_img = np.array(ori_img)
    pred_img = np.array(pred_img)
    target_img = np.array(target_img)



    ori_img = cv2.resize(ori_img, (256, 256), interpolation=cv2.INTER_AREA)
    target_img = cv2.resize(target_img, (256, 256), interpolation=cv2.INTER_AREA)
    pred_img = cv2.resize(pred_img, (256, 256), interpolation=cv2.INTER_AREA)

    ori_img = np.array(ori_img).astype('uint8')
    pred_img = np.array(pred_img).astype('uint8')
    target_img = np.array(target_img).astype('uint8')



    pred_cont = make_contours(pred_img)
    target_cont = make_contours(target_img)



    cv2.drawContours(ori_img, pred_cont, -1, (0, 0, 255), 2)  # pred(fill image : -1 )
    cv2.drawContours(ori_img, target_cont, -1, (255, 0, 0), 2)  # target(Positive : thick contours)

    colors = ['r', 'b']
    values = ['GT', 'pred']

    fig, ax = plt.subplots(1)
    ax.imshow(ori_img)
    plt.axis('off')
    plt.title(visual_name)

    # drow legend
    patches = [mpatches.Patch(color=colors[i], label="{}".format(values[i])) for i in range(len(values))]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    fig.savefig(path+fig_name)



def make_finddataset(folder_data, folder_mask,dataset,mode=False):
    """

    This is a function that looks for an image with the same name in the image.

    :param folder_data: The image folder you are looking for
    :param folder_mask: The image target you are looking for.
            len(folder_data) >= len(folder_mask)

    :return: folder_data[idx]

    """

    folder_data = np.array(sorted(folder_data))
    folder_mask = np.array(sorted(folder_mask))

    #print(folder_mask)
    # list name sort
    list_dataName = []
    for i in range(len(folder_data)):
        if dataset == 'SH_dataset' :
            # label mode
            if mode :
                data_name = folder_data[i].split('/')[-1].split('.')[0].split('_mask')[0]
            else:
                data_name = folder_data[i].split('/')[-1].split('.')[0]

            #print('dataname::',data_name)
            list_dataName.append(data_name)
        else:
            data_name = folder_data[i].split('/')[-1].split('.')[0]
            #print('dataname::', data_name)
            list_dataName.append(data_name)



    listdata = []
    for i in folder_mask:
        try:
            if dataset == 'JSRT_dataset':
                mask_name =i.split('_')[-2]
                #print('JSRT_dataset', mask_name)
            if dataset == 'MC_dataset':
                mask_name_ = i.split('_')[-4:-1]
                print('wwww',mask_name_)
                mask_name = ''
                for j in range(len(mask_name_)):
                    if j == len(mask_name_) - 1:
                        mask_name += str(mask_name_[j])
                    else:
                        mask_name += str(mask_name_[j]) + '_'
                print('MC_dataset name::', mask_name)
            if dataset == 'MC_modified_dataset':
                mask_name_ = i.split('_')[-3:]
                mask_name = ''
                for j in range(len(mask_name_)):
                    if j == len(mask_name_) - 1:
                        mask_name += str(mask_name_[j])
                    else:
                        mask_name += str(mask_name_[j]) + '_'
                mask_name =mask_name.split('.')[0]
                #print('MC_modified_dataset name::', mask_name)

            if dataset == 'SH_dataset':
                mask_name_ = i.split('_')[-3:]
                mask_name = ''
                for j in range(len(mask_name_)):
                    if j == len(mask_name_) - 1:
                        mask_name += str(mask_name_[j])
                    else:
                        mask_name += str(mask_name_[j]) + '_'
                mask_name = mask_name.split('.')[0]
                #print('SH_dataset', mask_name)

            list_dataName = np.array(list_dataName)
            idx = np.where(list_dataName == mask_name)[0][0]
            #print('idx', idx)
            listdata.append(folder_data[idx])
        except IndexError:
            continue

    return listdata



# visaulize result
dataset_list = ['SH_dataset','MC_modified_dataset']
model_list = ['All_Convolutional','UNetOriginal']


def visaulize_result(dataset_list,model_list):
    """

    This function visualizes the results of gt and pred.

    """
    pred_image = []
    ori_image = []
    target_image = []

    print(len(dataset_list))
    print(len(model_list))
    print('=====================')



    for i in range(len(dataset_list)):
        for j in range(len(model_list)):
            # serch data
            print(i)
            ori_image_paths = sorted(glob.glob("../dataset/{}/image/*.png".format(dataset_list[i])))
            target_image_paths = sorted(glob.glob("../dataset/{}/label/*.gif".format(dataset_list[i])))
            if len(target_image_paths) ==0:
                print('target image 00')
                target_image_paths = sorted(glob.glob("../dataset/{}/label/*.png".format(dataset_list[i])))
            pred_image_paths = sorted(glob.glob("../output_img/{}/{}/*.png".format(dataset_list[i],model_list[j])))


            path = "../visualize_result/{}/{}/".format(dataset_list[i], model_list[j])
            if not (os.path.isdir(path)):
                os.makedirs(os.path.join(path))
            ori=make_finddataset(ori_image_paths ,pred_image_paths,dataset_list[i])
            target = make_finddataset(target_image_paths,pred_image_paths,dataset_list[i],mode=True)
            pred = pred_image_paths


            # check
            if len(ori) == 0:
                print("+++++++++++")
                print(dataset_list[i],model_list[j])
            if len(target) == 0:
                print("+++++++++++")
                print(dataset_list[i], model_list[j])



            ori = sorted(ori)
            target = sorted(target)
            pred = sorted(pred)

            print(len(ori))
            print(len(target))
            print(len(pred))


            idx=[]
            for k in range(len(ori)):
                for l in range(len(pred)):
                    if ori[k].split('/')[-1].split('.')[0].split('_')[1] == pred[l].split('/')[-1].split('_')[2] and target[k].split('/')[-1].split('.')[0].split('_')[1] == pred[l].split('/')[-1].split('_')[2]:
                        #print(pred[j].split('/')[-1].split('_')[2])
                        idx.append(l)
                        print(l)

            pred = np.array(pred)
            pred = pred[idx]


            print(len(ori))
            print(len(pred))


            for z in range(len(pred)):
                make_visualize(ori[z],target[z],pred[z],path)


#visaulize_result(dataset_list,model_list)


#make_testCsv_list()
