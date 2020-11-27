# -*- coding: utf-8 -*-
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

'''

    Preprocess the fits files of different bands that belong to the same image.
    Compose a single image that can be used for creating batch for model input.
    Sample file name: 'frame-g-004263-4-0182.fits' (Temporary)
    
'''
class ProcessFits():
    def __init__(self, filePrefix = "frame-", fileSuffix = ".fits", directory="", fits_dir="fits", jpeg_dir="jpeg"):
        self.filePrefix = filePrefix
        self.fileSuffix = fileSuffix
        
        self.fits_dir = os.path.join(directory, fits_dir)
        self.jpeg_dir = os.path.join(directory, jpeg_dir)
    
    def getProcessedContent(self, fileName):
        hdu = fits.open(fileName)
        data = hdu[0].data
        return data
        
    def composeImage(self, commonFileName, bands=['r','g','i']):
        '''
            Prepare a single image data by combining the different bands of a fits.
        '''
        image = []
        for band in bands:
            filename = self.filePrefix + band + commonFileName + self.fileSuffix
            filename = os.path.join(self.fits_dir, filename)
            image.append(self.getProcessedContent(filename))
        image = np.array(image)
        image = self.standardize(image)
        return image
        
    def standardize(self, image):
        numerator = image - image.mean()
        denominator = max(image.std(), 1/np.sqrt(np.prod(image.shape)))
        return numerator/denominator
    
    def prepareImageList(self):
        '''
            Prepare a list of fits file names by retaining the part common to different bands of
            the same image.
        '''
        image_list = []
        with os.scandir(self.fits_dir) as fits_files:
            for fits in fits_files:
                commonFileName = fits.name.replace(self.filePrefix, '').replace(self.fileSuffix, '')
                commonFileName = commonFileName[1:]
                if commonFileName not in image_list:
                    image_list.append(commonFileName)
        return image_list
    
    def getJpegs(self):
        '''
            Returns the jpeg images whose corresponding fits files are present.
            Also removed the fits files from the list whose corresponding jpegs are not present.
        '''
        jpeg_list = []
        jpeg_content = []
        with os.scandir(self.jpeg_dir) as jpeg_files:
            for jpeg in jpeg_files:
                commonFileName = jpeg.name.replace(self.filePrefix, '').replace('irg', '').replace('.jpg', '')
                if commonFileName in self.image_list:
                    jpeg_list.append(commonFileName)
                    filename = os.path.join(jpeg)
                    image = cv2.imread(filename)
                    #image = image.reshape(3, image.shape[0], image.shape[1])
                    jpeg_content.append(image)
                    
        self.image_list = jpeg_list
        jpeg_content = np.array(jpeg_content)
        return jpeg_content
    
    def loadData(self,  bands=['r','g','i'], loadJpegs=False):
        '''
            Get a list of file names and load the content of the different bands corresponding to those 
            files.
        '''
        self.image_list = self.prepareImageList()
        data = []
        jpegs = []
        if loadJpegs:
            jpegs = self.getJpegs()
        for image in self.image_list:
            data.append(self.composeImage(image, bands))
        data = np.array(data)
        return data, jpegs
    
    def visualize(self, image):
        '''
            Visulaize a single image with three bands.
        '''
        rgb_image = make_lupton_rgb(image[0], image[1], image[2], stretch=1.5, Q=10)
        plt.imshow(rgb_image)
        
if __name__ == "__main__":
    p = ProcessFits(directory='data')
    f, j = p.loadData(loadJpegs=True)