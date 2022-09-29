"""
car type classification dataset
"""
#from ast import main
import os
#import pdb
#from tkinter.tix import MAIN
from PIL import Image
from torch.utils.data import Dataset
from utils import get_transform


DATAPATH = './CarType'
image_path = {}
image_label = {}


class CarTypeDataset(Dataset):
    def __init__(self, phase='train', resize=500):
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.resize = resize
        self.image_id = []
        self.num_classes = 9

        # get image path from images.txt
        with open(os.path.join(DATAPATH, 'images.txt')) as f:
            for line in f.readlines():
                id, path = line.strip().split(' ')
                image_path[id] = path

        # get image label from image_class_labels.txt
        with open(os.path.join(DATAPATH, 'image_class_labels.txt')) as f:
            for line in f.readlines():
                id, label = line.strip().split(' ')
                image_label[id] = int(label)

        # get train/test image id from train_test_split.txt
        with open(os.path.join(DATAPATH, 'train_test_split.txt')) as f:
            for line in f.readlines():
                image_id, is_training_image = line.strip().split(' ')
                is_training_image = int(is_training_image)

                if self.phase == 'train' and is_training_image:
                    self.image_id.append(image_id)
                if self.phase in ('val', 'test') and not is_training_image:
                    self.image_id.append(image_id)

        # transform
        self.transform = get_transform(self.resize, self.phase)

    def __getitem__(self, item):
        # get image id
        image_id = self.image_id[item]

        # image
        image = Image.open(os.path.join(DATAPATH, 'images', image_path[image_id])).convert('RGB') # (C, H, W)
        image = self.transform(image) # 改变大小

        # return image and label
        return image, image_label[image_id] # count begin from 0

    def __len__(self):
        return len(self.image_id)

    def get_all_image_path(self):
        all_image_path = []
        for i in range(len(self.image_id)):
            all_image_path.append(image_path[self.image_id[i]])
        return all_image_path


if __name__ == '__main__':
    cartype_dataset = CarTypeDataset('train', (448, 448))
    print(len(cartype_dataset))
    for i in range(0, 10):
        image, label = cartype_dataset[i]
        print(image.shape, label)