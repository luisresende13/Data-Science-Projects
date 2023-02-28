import cv2, pathlib, numpy as np, pandas as pd
from IPython.display import clear_output as co

def load_nested_images_from_folder(folder, ext='.jpg', first=None):
    path_list = [img for img in pathlib.Path(folder).rglob("*") if str(img).endswith(ext)]
    if first is None: first = len(path_list)
    frames = []
    for i, path in enumerate(path_list[:first]):
        if i % 10 == 0: co(True); print(f'Images loaded: {i}/{first}')
        frame = cv2.imread(str(path))
        frames.append(frame)
    co(True); print(f'Done! Loaded total of {first} images.')
    return frames

class img:
    
    def avg_prct_diff(img1, img2):
        if type(img1) is list: return [img.avg_prct_diff(im1, im2) for im1, im2 in zip(img1, img2)]
        return (np.abs(img1.astype(int) - img2.astype(int)) / 255).mean()
    
    def isdiff(img1, img2, p=0.1):  # if avg_prct_diff is more than 'p', images are considered different    
        if type(img1) is list: return [img.isdiff(im1, im2, p) for im1, im2 in zip(img1, img2)]
        return img.avg_prct_diff(img1, img2) >= p

    # if avg_prct_diff is more than 'p', images are considered different    
    def isnotdiff(img1, img2, p=0.1):
        if type(img1) is list: return [not isdiff for isdiff in img.isdiff(img1, img2, p)]
        return not img.isdiff(img1, img2, p)
    
class similarity_classifier:

    def __init__(self, baseimgs=[], p=0.2):
        self.p = p
        self.nbases = len(baseimgs)
        self.baseimgs = [cv2.imread(path) for path in baseimgs]
        self.baseids = baseimgs
    
    def predict_any(self, image):
        if type(image) is list: return list(map(self.predict_any, image))
        return np.any(self.predict(image))
    
    def predict(self, image):
        if type(image) is list: return list(map(self.predict, image))
        return [isnotdiff for isnotdiff in img.isnotdiff(self.baseimgs, self.nbases*[image], p=self.p)]
    
    def diffs(self, image):
        if type(image) is list: return list(map(self.diffs, image))
        return img.avg_prct_diff(self.baseimgs, self.nbases*[image])

    def min_diff(self, image):
        if type(image) is list: return list(map(self.min_diff, image))
        return np.min(self.diffs(image))