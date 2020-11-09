# -*- coding: utf-8 -*-
from astropy.io import fits
import numpy as np

'''

    Preprocess the fits files of different bands that belong to the same image.
    Compose a single image that can be used for creating batch for model input.
    Sample file name: 'frame-g-004263-4-0182.fits' (Temporary)
    
'''
class ProcessFits():
    def __init__(self, filePrefix = "frame-", fileSuffix = ".fits", directory=""):
        self.filePrefix = directory+filePrefix
        self.fileSuffix = fileSuffix
    
    def getProcessedContent(self, fileName):
        hdu = fits.open(fileName)
        data = hdu[0].data
        return data
        
    def composeImage(self, commonFileName, bands=['r','g','i']):
        image = []
        for band in bands:
            filename = self.filePrefix + band + '-' + commonFileName + self.fileSuffix
            image.append(self.getProcessedContent(filename))
        image = np.array(image)
        image = self.standardize(image)
        return image
        
    def standardize(self, image):
        numerator = image - image.mean()
        denominator = max(image.std(), 1/np.sqrt(np.prod(image.shape)))
        return numerator/denominator