import numpy as np
import cv2 as cv

class HistogramClassifier:
    
    def __init__(self, threshold:float=0.6, num_of_bins:int=256, range:list=[0,256]):
        self.threshold = threshold; self.num_of_bins = num_of_bins; self.range = range

    def is_histogram_clustered(self, image):
        if type(image) is list: return list(map(self.is_histogram_clustered, image))
        if len(image.shape) == 3: image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return self.histogram_max_prct(image) >= self.threshold

    def histogram_max_prct(self, image):
        if type(image) is list: return list(map(self.histogram_max_prct, image))
        hist = np.histogram(image, bins=self.num_of_bins, range=self.range)
        return hist[0].max() / hist[0].sum()