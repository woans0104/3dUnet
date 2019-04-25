
import numpy as np
import glob
import os
import pandas as pd
from PIL import Image


def save_log(filename, image, label):
    # fullpath=path+str(filename)

    if type(image) != list and type(image) != np.ndarray:
        image = [image]
        label = [label]

    data = {'img': image, 'lbl': label}
    f = pd.DataFrame(data=data, columns=['img', 'lbl'])


    f.to_csv(filename, index=False)

# Selects non-255 values ​​from the dataset.
def find_label_(label):
    labels1 = Image.open(label)
    labels2 = np.array(labels1)
    if set(labels2.reshape(-1)) == {0,1}:
        return label


def make_csv():
    #dirname = '/data2/brain_ct/trainvalid_3d'
    dirname = '/data2/brain_ct/test_3d'

    filenames = os.listdir(dirname)

    image = []
    label = []
    image_list = []
    label_list = []
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        image_folder = sorted(glob.glob(full_filename+'/*'))
        """
        image_list = []
        label_list = []
        """

        for i in image_folder:
            ext = os.path.splitext(i)[-1]
            if ext == '.gif':
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

    filename = './find_testdataset.csv'
    save_log(filename,image_list,label_list)

    return image,label




