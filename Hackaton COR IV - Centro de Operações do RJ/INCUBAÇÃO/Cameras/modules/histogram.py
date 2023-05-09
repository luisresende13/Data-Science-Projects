import numpy as np
import cv2 as cv
import modules.util as util
from matplotlib import pyplot as plt
# import io
# from PIL import Image

def is_histogram_clustered(image, num_of_bins:int = 256, range=[0,256], threshold:float=0.73):
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    hist, _ = np.histogram( image, bins=num_of_bins, range=range)

    return True if hist.max()/hist.sum() >= threshold else False

def histogram(image, hist_size=[256], range=[0,256]):
    return cv.calcHist([image], [0], None, hist_size, range)

def show_histogram(image, num_of_bins:int = 256, range=[0,256]):
    plt.hist(image.ravel(),num_of_bins,range)
    plt.show()

def normalize(image, alpha=0, beta=255):
    image_normalized = np.zeros(image.shape)
    image_normalized = cv.normalize(image, image_normalized, alpha, beta, cv.NORM_MINMAX)
    return image_normalized

def histogram_3_channels(src, hist_w:int=512, hist_h:int=400):
    # https://docs.opencv.org/4.4.0/d8/dbc/tutorial_histogram_calculation.html#gsc.tab=0
    bgr_planes = cv.split(src)
    histSize = 256
    histRange = (0, 256) # the upper boundary is exclusive
    accumulate = False

    
    b_hist = cv.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
    g_hist = cv.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
    r_hist = cv.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)

    bin_w = int(round( hist_w/histSize ))
    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

    cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    cv.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    cv.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)

    for i in range(1, histSize):
        cv.line(histImage, ( bin_w*(i-1), hist_h - int(round(b_hist[i-1])) ),
                ( bin_w*(i), hist_h - int(round(b_hist[i])) ),
                ( 255, 0, 0), thickness=2)
        cv.line(histImage, ( bin_w*(i-1), hist_h - int(round(g_hist[i-1])) ),
                ( bin_w*(i), hist_h - int(round(g_hist[i])) ),
                ( 0, 255, 0), thickness=2)
        cv.line(histImage, ( bin_w*(i-1), hist_h - int(round(r_hist[i-1])) ),
                ( bin_w*(i), hist_h - int(round(r_hist[i])) ),
                ( 0, 0, 255), thickness=2)
        
    return histImage

def get_histogram_file(image, num_of_bins=256, range=[0,256], title=''):
    '''
    Saves the histogram graph from 'image' as a file
    '''
    hist = histogram(image, [num_of_bins], range)
    return util.get_graph_file(hist, title)

def old_get_histogram_file(image, num_of_bins=256, range=[0,256], title=''):
    import io
    from PIL import Image
    
    plt.hist(image.ravel(), num_of_bins, range)
    buf = io.BytesIO()
    plt.title(title)
    plt.savefig(buf, format='png')
    plt.clf()
    buf.seek(0)
    return cv.UMat(np.array(Image.open(buf), dtype=np.uint8)) #  Needed to convert PIL format to UMat