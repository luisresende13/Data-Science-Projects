from datetime import datetime as dt
import pytz; tzbr = pytz.timezone('Brazil/East')
from modules.googlecloudstorage import GCS
from tempfile import NamedTemporaryFile
import cv2, multiprocessing, numpy as np
from modules.image_similarity import similarity_classifier

def now(fmt="%Y-%m-%d %H-%M-%S"):
    return dt.now(tzbr).strftime(fmt)

# google cloud storage params

gcs = GCS('../../../../Apps/Python/bolsao-api/credentials/pluvia-360323-35cd376d5958.json') # YOUR GOOGLE CLOUD SERVICE ACCOUNT JSON FILE PATH
bucket_name = 'city-camera-images' # YOUR GOOGLE CLOUD STORAGE BUCKET NAME

# Skip invalid frames params

baseimgs = ['Gabaritos/cam.jpg', 'Gabaritos/dark.jpg']

p = 0.05
clf = similarity_classifier(baseimgs, p)

skip_methods= {
    'histogram': clf.is_histogram_clustered,
    'avg_prct_diff': clf.predict_any
}

# File handling methods

def upload_frames(frames, path):
    for filename, frame in frames:
        blob_name = f'{path}/{filename}.jpg'
        gcs.upload_from_file(frame, blob_name, bucket_name, 'image/jpeg')

def upload_video(frames, blob_name):
    with NamedTemporaryFile() as temp:
        tname = f"{temp.name}.mp4"
        write_video(frames, tname, fps=3, codec='mp4v')
        gcs.upload_from_filename(tname, blob_name, bucket_name, 'video/mp4')

def write_video(frames, path, shape='auto', fps=3, codec='mp4v'):
    if shape == 'auto': height, width, _ = frames[0].shape; shape = (width, height) 
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(path, fourcc, fps, shape)
    for frame in frames: writer.write(frame)
    cv2.destroyAllWindows(); writer.release()
    
class recorder:

    url = 'http://187.111.99.18:9004/?CODE={}'

    def __init__(self, folder='.', n_frames=None, skip_method=None, skip_max=None, saveas='video', retries=10):
        self.folder = folder
        self.saveas = saveas
        self.n_frames = n_frames
        self.skip_method = skip_method
        self.skip_max = skip_max
        self.retries = retries
        if skip_method is not None:
            self.is_frame_invalid = skip_methods[skip_method]
            
    def capture_frame(self, cap, url): # retries could be implemented here
        success, frame = cap.read(); r = 0
        while not success and r < self.retries:
            cap = cv2.VideoCapture(url)
            success, frame = cap.read(); r += 1
        return cap, frame # frame is None if retries are exceeded
    
    def capture(self, code):
        url = self.url.format(code)
        cap = cv2.VideoCapture(url)
        skipped, appended, frames = 0, 0, []
        while(True):
            cap, frame = self.capture_frame(cap, url)
            if frame is None:
                print(f'CONNECTION LOST. CODE {code}. MAX RETRIES ({self.retries}) EXCEEDED.')
                break
            if self.skip_method is not None:
                if self.is_frame_invalid(frame):
                    skipped +=1
                    if self.skip_max is not None:
                        if skipped == self.skip_max:
                            print(f'CAPTURE STOPPED. CODE {code}. MAX SKIPPED FRAMES ({self.skip_max}) EXCEEDED.')
                            break
                    continue
            filename = f'CODE{code} {now("%Y-%m-%d %H-%M-%S-%f")[:-5]}'  # [:-5] keeps only the first microsecond decimal
            frames.append([filename, frame]); appended += 1
            if self.n_frames is not None:
                if appended >= self.n_frames: break
        return frames
    
    def record(self, code):
        path = f'{self.folder}/{code}'
        stamp = now()
        frames = self.capture(code)
        if not len(frames):
            print(f'FAILED TO RECORD. CODE {code}, AT {stamp}.'); return
        if self.saveas == 'image':
            upload_frames(frames, f'{path}/{stamp}')
        elif self.saveas == 'video':
            frames_arr = [frame for _, frame in frames]
            blob = f'{path}/CODE{code} {stamp}.mp4'
            upload_video(frames_arr, blob)

    def record_many(self, codes, workers='auto'):
        if not len(codes): print(f'TASK SKIPPED. FOLDER {self.folder}, AT {now("%Y-%m-%d %X")}. EMPTY CODE LIST PROVIDED.'); return
        if workers == 'auto': workers = len(codes)
        pool = multiprocessing.Pool(processes=workers)
        codes = [(code,) for code in codes]
        stamp = now(); pool.starmap(self.record, codes)
        self.report_finished(stamp)
        return

    def report_finished(self, stamp):
        print(f"Job started at'{stamp}' finished at '{now()}'.")