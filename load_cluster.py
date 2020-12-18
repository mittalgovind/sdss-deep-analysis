# -*- coding: utf-8 -*-
import pickle
from data_loader import DataLoader
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import cv2
import numpy as np

cluster_file = 'clusters_finetune'

f = open(cluster_file, 'rb')
clusters = pickle.load(f)[-1]

loader = DataLoader()
files = loader.load_data(root='extracted_objects', n_objects=10000)

objects_per_cluster = 4

clusters = [cluster[:objects_per_cluster] for cluster in clusters]

cluster_content = []
for cluster in clusters:
    content = []
    for object_id in cluster:
        content.append(files[object_id])
    cluster_content.append(content)
    
grid_size = int(objects_per_cluster**(1/2))
for i, cluster in enumerate(cluster_content):
    fig = plt.figure(figsize=(4.0, 4.0))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(grid_size, grid_size),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    image_data = []
    for img in cluster:
        image = cv2.imread(img)
        image_data.append(image)
    image_data = np.array(image_data)
    
    for ax, im in zip(grid, image_data):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
    plt.savefig('cluster_finetune' + str(i) + '.jpg')
