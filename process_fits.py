# -*- coding: utf-8 -*-
from astropy.io import fits
import numpy as np
import os

'''

    Preprocess the fits files of different bands that belong to the same image.
    Compose a single image that can be used for creating batch for model input.
    Sample file name: 'frame-g-004263-4-0182.fits' (Temporary)
    
'''
class ProcessFits():
    def __init__(self, filePrefix = "frame-", fileSuffix = ".fits", directory=""):
        self.directory = directory
        self.filePrefix = filePrefix
        self.fileSuffix = fileSuffix
    
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
            filename = os.path.join(self.directory, filename)
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
        with os.scandir(self.directory) as fits_files:
            for fits in fits_files:
                commonFileName = fits.name.replace(self.filePrefix, '').replace(self.fileSuffix, '')
                commonFileName = commonFileName[1:]
                if commonFileName not in image_list:
                    image_list.append(commonFileName)
        return image_list
    
    def loadData(self,  bands=['r','g','i']):
        '''
            Get a list of file names and load the content of the different bands corresponding to those 
            files.
        '''
        image_list = self.prepareImageList()
        data = []
        for image in image_list:
            data.append(self.composeImage(image, bands))
        return data