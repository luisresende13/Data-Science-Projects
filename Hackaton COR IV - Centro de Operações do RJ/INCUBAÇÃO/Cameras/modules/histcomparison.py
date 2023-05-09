import cv2 as cv
import modules.histogram as hist

class SimpleHDA:
    def __init__(self, id, background, mask, flood_reference):
        self.id = id
        self.background = background
        self.reference = flood_reference
        # Mask must be a 2-dimensional array with 1 channel
        if len(mask.shape) > 2:
            self.mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        else:
            self.mask = mask

    def preprocessing(self, img):
        # LUV
        img_luv = cv.cvtColor(img, cv.COLOR_BGR2LUV)
        # Normalize
        img_luv[:,:,0] = hist.normalize(img_luv[:,:,0])
        # Grayscale
        img_gray = cv.cvtColor(img_luv, cv.COLOR_LUV2BGR)
        img_gray = cv.cvtColor(img_gray, cv.COLOR_BGR2GRAY)
        # Apply mask
        return img_gray*self.mask
    
    def img_processing(self, img):
        return self.preprocessing(img) - self.preprocessing(self.background)
    
    def dissimilarity(self, img, compare_method = cv.HISTCMP_BHATTACHARYYA):
        '''
        parameters:
            img - Image of same camera and shape as 'background' image
            compare_method - Histogram comparison method as defined in cv::HistCompMethods.
                             Bhattacharyya distance as default method.          
        '''
        reference_hist = hist.histogram(self.img_processing(self.reference))
        img_hist = hist.histogram(self.img_processing(img))
        return round(cv.compareHist(reference_hist, img_hist, compare_method),4)

    def predict(self, img, compare_method = cv.HISTCMP_BHATTACHARYYA):

        if self.dissimilarity(img, compare_method) <=  0.2:
            return 'acumulo'
        return 'normalidade'
    

class HistogramDissimilarityAnalysis:
    
    def __init__(self, id, background, mask, reference_day, reference_night, reference_puddle, reference_flood):
        self.id = id
        self.background = background
        
        self.reference = {}
        self.reference['day'] = reference_day
        self.reference['night'] = reference_night
        self.reference['puddle'] = reference_puddle
        self.reference['flood'] = reference_flood
        
        # Mask must be a 2-dimensional array with 1 channel
        if len(mask.shape) > 2:
            self.mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        else:
            self.mask = mask
    
    def preprocessing(self, img):
        # LUV
        img_luv = cv.cvtColor(img, cv.COLOR_BGR2LUV)
        # Normalize
        img_luv[:,:,0] = hist.normalize(img_luv[:,:,0])
        # Grayscale
        img_gray = cv.cvtColor(img_luv, cv.COLOR_LUV2BGR)
        img_gray = cv.cvtColor(img_gray, cv.COLOR_BGR2GRAY)
        # Apply mask
        return img_gray*self.mask
    
    def img_processing(self, img):
        return self.preprocessing(img) - self.preprocessing(self.background)
    
    def dissimilarity(self, img, reference_img = 'flood', compare_method = cv.HISTCMP_BHATTACHARYYA):
        '''
        parameters:
            img - Image of same camera and shape as 'background' image
            reference_img - 'day','night','puddle' or 'flood'. 'flood' as default
            compare_method - Histogram comparison method as defined in cv::HistCompMethods.
                             Bhattacharyya distance as default method.          
        '''
        reference_hist = hist.histogram(self.img_processing(self.reference[reference_img]))
        img_hist = hist.histogram(self.img_processing(img))
        return round(cv.compareHist(reference_hist, img_hist, compare_method),4)
    
    def dissimilarity_dict(self, img, compare_method = cv.HISTCMP_BHATTACHARYYA):
        '''
        parameters:
            img - Image of same camera and shape as 'background' image
            compare_method - Histogram comparison method as defined in cv::HistCompMethods.
                             Bhattacharyya distance as default method.          
        '''
        dict_out = {}
        for key in self.reference:
            dict_out[key] = self.dissimilarity(img, key, compare_method=compare_method)
        return dict_out

    def predict(self, img, compare_method = cv.HISTCMP_BHATTACHARYYA):
        diss_dict = self.dissimilarity_dict(img, compare_method)
        min_val_key = min(diss_dict, key=diss_dict.get)

        if min_val_key == 'puddle':
            return 'acúmulo'
        if min_val_key == 'flood':
            return 'acúmulo'

        return 'normalidade'