# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
from process_fits import ProcessFits
from astropy.visualization import make_lupton_rgb

class Segmentation:
    def getBoundingBoxes(self, image):
        # thresholding
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #blur = cv2.GaussianBlur(gray,(5,5),0)
        threshold_value, threshold_image = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # get contours
        contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def visualiseBoundingBox(self, image, contours):
        # bounding boxes
        original = image
        for cont in contours:
            x,y,w,h = cv2.boundingRect(cont)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        
        cv2.imshow('image', image)
        cv2.waitKey()
        
    def compareImages(self, *images):
        for i, image in enumerate(images):
            plt.subplot(256,256,i*256+1),plt.imshow(image)
        plt.show()
    
    def fitsToRGB(self, fits_image):
        rgb_image = make_lupton_rgb(fits_image[0], fits_image[1], fits_image[2], stretch=1.5, Q=10)
        return rgb_image
    
    def getSegments(self, image, countours):
        segments = []
        for cont in countours:
            x,y,w,h = cv2.boundingRect(cont)
            segments.append(image[:, x:x+w, y:y+h])
        return segments
        
    def processSegmentation(self):
        processFits = ProcessFits(directory="sample_data")
        fits_images = processFits.loadData()
        
        rgb_images = [self.fitsToRGB(image) for image in fits_images]
        
        contours = self.getBoundingBoxes(rgb_images[0])
        self.visualiseBoundingBox(rgb_images[0], contours)
        
        return contours
