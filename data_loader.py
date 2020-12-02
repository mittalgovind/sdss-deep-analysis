# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader:
    def load_data(self, root='objects', n_objects=None, split=0.8, augment=True, seed=69, shuffle=True):
        #load
        objects = []
        count=0
        for dirpath, dirnames, filenames in os.walk(root):
            for file in filenames:
                if (n_objects is not None) and (count==n_objects):
                    break
                count += 1
                file = os.path.join(dirpath, file)
                image = cv2.imread(file)
                objects.append(image)
        objects = np.array(objects)
        
        # data augment
        
        # split
        train_data, val_data = train_test_split(objects, test_size=(1-split), random_state=seed, shuffle=shuffle)
        
        return train_data, val_data
    
if __name__ == '__main__':
    loader = DataLoader()
    train_data, val_data = loader.load_data(root='sample_objects')