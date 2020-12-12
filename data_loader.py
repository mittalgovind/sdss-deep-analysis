# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import re
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self):
        self.regex1 = re.compile(r'[a-zA-Z0-9\\_]*frame[0-9-]+\\*')
        self.regex2 = re.compile(r'_[a-zA-Z0-9.]+')
    
    def resolution(self, filename):
        filename = re.sub(self.regex1, '', filename)
        filename = re.sub(self.regex2, '', filename)
        resolution = int(filename)
        return resolution
    
    def select_top_objects(self, files, n_objects):
        files.sort(key = self.resolution, reverse=True)
        files = files[:n_objects]
        return files
    
    def load_data(self, root='objects', n_objects=None, split=0.8, augment=True, seed=69, shuffle=True):
        #load
        objects = []
        files = []
        for dirpath, dirnames, filenames in os.walk(root):
            for file in filenames:
                file = os.path.join(dirpath, file)
                files.append(file)
        if n_objects is not None:
            files = self.select_top_objects(files, n_objects)
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