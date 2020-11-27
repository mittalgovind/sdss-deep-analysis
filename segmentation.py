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
    
    def getSegmentedContents(self, image_list, contour_list, fits):
        contents = []
        for image, contour in zip(image_list, contour_list):
            for obj in contour:
                x,y,w,h = cv2.boundingRect(obj)
                if w*h >= 100:
                    if fits:
                        contents.append(image[:, x:x+w, y:y+h])
                    else:
                        contents.append(image[x:x+w, y:y+h, :])
        return contents
        
    def processSegmentation(self, fromJpeg=True, mapToFits=True, directory='data'):
        processFits = ProcessFits(directory=directory)
        fits_images, jpeg_images = processFits.loadData(loadJpegs=fromJpeg)
        
        segmented_contents = []
        if fromJpeg:
            contours = [self.getBoundingBoxes(img) for img in jpeg_images]
        else:
            rgb_images = [self.fitsToRGB(image) for image in fits_images]
            contours = [self.getBoundingBoxes(img) for img in rgb_images]
        
        if mapToFits:
            segmented_contents = self.getSegmentedContents(fits_images, contours, True)
        else:
            segmented_contents = self.getSegmentedContents(jpeg_images, contours, False)
        
        #contours = self.getBoundingBoxes(rgb_images[0])
        #self.visualiseBoundingBox(rgb_images[0], contours)
        
        return segmented_contents
    
if __name__ == "__main__":
    segment = Segmentation()
    contents = segment.processSegmentation(mapToFits=False)
