import mimetypes
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import io
from PIL import Image
import os
from datetime import datetime
from pytz import timezone

def current_time():
    return datetime.now(timezone("Brazil/East")).strftime("%Y-%m-%d_%H-%M-%S")

def get_last_char(string:str):
    return string[len(string)-1]

def get_string_after_delimiter(string:str, delimiter:str):
    '''
    if the string ends in the delimiter, or if there is no delimiter,
    returns the original string without the delimiter
    '''
    start, delim, end = string.rpartition(delimiter)
    return end if (delim and end) else start

def file_has_extension(filename:str):
    extension = filename.split('.')[-1]
    # print(extension)
    return True if len(extension) > 0 else False

def get_file_extension(filename:str):
    file_extension = filename.split('.')[-1]
    # print(file_extension)
    return file_extension if len(file_extension) > 0 else mimetypes.guess_extension(filename)

def get_graph_file(image, title:str=''):
    plt.plot(image)
    buf = io.BytesIO()
    plt.title(title)
    plt.savefig(buf, format='png')
    plt.clf()
    buf.seek(0)
    return cv.UMat(np.array(Image.open(buf), dtype=np.uint8)) #  Needed to convert PIL format to UMat

def get_filenames_from_folder(folder:str):
    file_list = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_list.append(os.path.join(root,file).replace('\\','/'))
    return file_list