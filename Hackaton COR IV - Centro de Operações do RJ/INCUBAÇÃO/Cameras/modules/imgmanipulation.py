import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from modules.util import get_filenames_from_folder

def resize(image, prct):
    rows, cols = image.shape
    new_rows = int(prct*rows)
    new_cols = int(prct*cols)
    new_dim = (new_cols, new_rows)
    new_image = cv.resize(image, new_dim)
    return new_image

def show_images(images):
    #fig = plt.figure(figsize=( int(len(images)/2), len(images) - int(len(images)/2)))
    k = len(images)
    fig = plt.figure(figsize=(k,k))
    i=1
    for image in images:
        ax = fig.add_subplot(int(len(images)/2),2,i)
        ax.imshow(image, cmap='gray')
        ax.title.set_text(str(i))
        i += 1
    plt.show()

def show_image(image, title=""):
    plt.imshow(image)
    plt.title(title)
    plt.show()

def show_image_and_channels(img):
    # plt.imshow(img)
    # plt.title("Original image")
    # plt.show()
    show_image(img, "Original image")
    for channel in range(img.shape[2]):
        # plt.imshow(img[:,:,channel])
        # plt.title(f"Channel {channel}")
        # plt.show()
        show_image(img[:,:,channel], f"Channel {channel}")
        
def mean_frame(img_list):
    return (np.array(img_list, dtype='int').sum(axis=0) / len(img_list)).round().astype(int)

def read_images_from_folder(folder:str):
    images = []
    print(f'Reading folder {folder}')
    filenames = get_filenames_from_folder(folder)
    for filename in filenames:
        image = cv.imread(filename)
        if image is not None:
            images.append(image)
    return images

def convert_frames_to_video(list_of_images, output_name:str="output", extension:str=".mp4", codec:str="mp4v", fps=3):
    '''
    Converts list of frames to video and saves in hard drive
    images must have the same number of channels
    '''
    # height, width, _ = list_of_images[0].shape
    height = list_of_images[0].shape[0]
    width = list_of_images[0].shape[1]
    
    is_color_image = True
    if len(list_of_images[0].shape) == 2:
        is_color_image = False
    # decidir quantos frames do feed armazenar antes de converter para video e fazer upload 
    
    video = cv.VideoWriter(f'{output_name}{extension}',cv.VideoWriter_fourcc(*codec), fps, (width, height), is_color_image)

    for image in list_of_images:
        video.write(image)
    
    cv.destroyAllWindows()
    video.release()