# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import re
from sklearn.model_selection import train_test_split
import torch.utils.data as data


class DataLoader(data.Dataset):
    def __init__(self, root='sample_objects', n_objects=None, transform=None):
        self.regex1 = re.compile(r'[a-zA-Z0-9\\_]*frame[0-9-]+\\*')
        self.regex2 = re.compile(r'_[a-zA-Z0-9.]+')

        self.files = self.load_data(root=root, n_objects=n_objects, split=1.0)
        if n_objects is None:
            n_objects = len(self.files)

        self.transform = transform
        self.sub_classes = None
        self.indexes = np.arange(n_objects)
        self.subset_indexes = None

    def __getitem__(self, ind):
        img = cv2.imread(self.files[ind])

        # apply transformation
        if self.transform is not None:
            img = self.transform(img)

        # id of cluster
        sub_class = -100
        if self.sub_classes is not None:
            sub_class = self.sub_classes[ind]

        return img, sub_class

    def __len__(self):
        if self.subset_indexes is not None:
            return len(self.subset_indexes)
        return len(self.indexes)

    def resolution(self, filename):
        filename = re.sub(self.regex1, '', filename)
        filename = re.sub(self.regex2, '', filename)
        resolution = int(filename)
        return resolution

    def select_top_objects(self, files, n_objects):
        files.sort(key=self.resolution, reverse=True)
        files = files[:n_objects]
        return files

    def load_data(self, root='objects', n_objects=None, split=0.8,
                  augment=True, seed=69, shuffle=True):
        # load
        objects = []
        files = []
        for dirpath, dirnames, filenames in os.walk(root):
            for file in filenames:
                file = os.path.join(dirpath, file)
                files.append(file)
        if n_objects is not None:
            files = self.select_top_objects(files, n_objects)
        return files


'''

for file in files:
            image = cv2.imread(file)
            objects.append(image)
        objects = np.array(objects)
        
        # data augment
        
        # split
        train_data, val_data = train_test_split(objects, test_size=(1-split), random_state=seed, shuffle=shuffle)
        
        return train_data, val_data
    
if __name__ == '__main__':
    loader = DataLoader()
    train_data, val_data = loader.load_data(root='sample_objects', n_objects=10)

'''
