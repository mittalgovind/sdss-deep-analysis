# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from process_fits import ProcessFits
from data_loader import DataLoader
import cv2
import numpy as np

loader = DataLoader()
files = loader.load_data(root='extracted_objects', n_objects=25)

image_data = []
for img in files:
    image = cv2.imread(img)
    image_data.append(image)
image_data = np.array(image_data)

p = ProcessFits(directory='')
fits, image_data, files = p.loadData(loadJpegs=True, loadFits=False)

fig = plt.figure(figsize=(16., 16.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(5, 5),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

for ax, im in zip(grid, image_data):
    # Iterating over the grid returns the Axes.
    ax.imshow(im)

plt.show()
