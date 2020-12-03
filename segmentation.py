# -*- coding: utf-8 -*-
import cv2
from matplotlib import pyplot as plt
from process_fits import ProcessFits
from astropy.visualization import make_lupton_rgb
import os

'''
    Extract objects from images of patch of sky.
'''
class Segmentation:
    def __init__(self, thresholdObjectArea=400):
        # thresholdObjectArea determines the minimum area (in pixels) for an object to be selected.
        self.thresholdObjectArea = thresholdObjectArea
        self.log_index = 0
    
    def getBoundingBoxes(self, image):
        # thresholding
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #blur = cv2.GaussianBlur(gray,(5,5),0)
        threshold_value, threshold_image = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # get contours
        contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print("Contours for image: ", self.log_index)
        self.log_index += 1
        return contours
    
    def visualiseBoundingBox(self, image, contours):
        # bounding boxes
        for cont in contours:
            x,y,w,h = cv2.boundingRect(cont)
            if w*h >= self.thresholdObjectArea:
                cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        
        cv2.imshow('image', image)
        cv2.waitKey()
        
    def compareImages(self, *images):
        fig = plt.figure(figsize=(10, 10))
        rows = len(images)//5
        for i, image in enumerate(images):
            fig.add_subplot(rows, 5, i)
            plt.imshow(image)
        plt.show()
    
    def fitsToRGB(self, fits_image):
        rgb_image = make_lupton_rgb(fits_image[0], fits_image[1], fits_image[2], stretch=1.5, Q=10)
        return rgb_image
    
    def getSegmentedContents(self, image_list, contour_list, file_names, fits):
        contents = {}
        for image, contour, file in zip(image_list, contour_list, file_names):
            contents[file] = []
            for obj in contour:
                x,y,w,h = cv2.boundingRect(obj)
                if w*h >= self.thresholdObjectArea:
                    if fits:
                        contents[file].append((image[:, y:y+h, x:x+w], w*h))
                    else:
                        contents[file].append((image[y:y+h, x:x+w, :], w*h))
        return contents
        
    def processSegmentation(self, fromJpeg=True, mapToFits=True, directory='data'):
        '''
            This method loads the images of patch of sky, detects and returns a list of objects.
            If fromJpeg is set then object bounding boxes are detected from jpeg images.
            If mapToFits is set then detected bounding boxes are used to get contents from fits files.
        '''
        print("Loading patches of sky...")
        processFits = ProcessFits(directory=directory)
        fits_images, jpeg_images, file_names = processFits.loadData(loadJpegs=fromJpeg, loadFits=mapToFits)
        
        segmented_contents = []
        print("Getting bounding boxes...")
        if fromJpeg:
            contours = [self.getBoundingBoxes(img) for img in jpeg_images]
        else:
            rgb_images = [self.fitsToRGB(image) for image in fits_images]
            contours = [self.getBoundingBoxes(img) for img in rgb_images]
        
        print("Extracting contents...")
        if mapToFits:
            segmented_contents = self.getSegmentedContents(fits_images, contours, file_names, True)
        else:
            segmented_contents = self.getSegmentedContents(jpeg_images, contours, file_names, False)
        
        return segmented_contents
    
    def standardScaler(self, objects):
        '''
            Bring objects to a common scale (largest scale amoung the images to retain information). 
        '''
        #greatest = 0
        print("Scaling objects...")
        standard_dim = (128, 128)
        '''for obj in objects:
            w, h  = obj.shape[0], obj.shape[1]
            if w*h > greatest:
                greatest = w*h
                standard_dim = (w,h)'''
        for image in objects.keys():
            length = len(objects[image])
            for i in range(length):
                objects[image][i] = (cv2.resize(objects[image][i][0], standard_dim), objects[image][i][1])
        return objects
    
    def saveObjects(self, objects, folder='extracted_objects'):
        print("Saving objects...")
        if not os.path.exists(folder):
            os.makedirs(folder)
        for image in objects.keys():
            sub_folder = 'frame' + image
            sub_folder = os.path.join(folder, sub_folder)
            if not os.path.exists(sub_folder):
                os.makedirs(sub_folder)
            for i, obj in enumerate(objects[image]):
                filename = str(obj[1]) + '_' + str(i) + '.jpg'
                filename = os.path.join(sub_folder, filename)
                cv2.imwrite(filename, obj[0])
    
if __name__ == "__main__":
    thresholdObjectArea=625
    segment = Segmentation(thresholdObjectArea)
    objects = segment.processSegmentation(mapToFits=False)
    std_obj = segment.standardScaler(objects)
    segment.saveObjects(std_obj)
    print("Done!")