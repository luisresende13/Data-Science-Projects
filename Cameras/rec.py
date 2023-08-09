# Import modules

import numpy as np, cv2
from tempfile import NamedTemporaryFile
from datetime import datetime as dt
import pytz; tzbr = pytz.timezone('Brazil/East')

# instantiated objects and methods

from modules.util import skip_methods, gcs, bucket_name

# time tracking

def now(fmt="%Y-%m-%d %H-%M-%S"):
    return dt.now(tzbr).strftime(fmt)

# File and storage handling methods

class Video:
    
    def __init__(self, shape=(854, 480), fps=3, codec='mp4v'):
        self.shape=shape; self.fps=fps; self.codec=codec
        
    def writer(self, path):
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        return cv2.VideoWriter(path, fourcc, self.fps, self.shape)
    
    def close(writer):
        cv2.destroyAllWindows(); writer.release()
    
def write_video(frames, path, shape='auto', fps=3, codec='mp4v'):
    if shape == 'auto': height, width, _ = frames[0].shape; shape = (width, height) 
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(path, fourcc, fps, shape)
    for frame in frames: writer.write(frame)
    cv2.destroyAllWindows(); writer.release()

def upload_video(frames, blob_name):
    with NamedTemporaryFile() as temp:
        tname = f"{temp.name}.mp4"
        write_video(frames, tname, fps=3, codec='mp4v')
        gcs.upload_from_filename(tname, blob_name, bucket_name, 'video/mp4')

def upload_frames(frames, path):
    for filename, frame in frames:
        blob_name = f'{path}/{filename}.jpg'
        gcs.upload_from_file(frame, blob_name, bucket_name, 'image/jpeg')

# City camera video recorder class

class Recorder:

    url = 'http://187.111.99.18:9004/?CODE={}'

    def __init__(self, folder='.', n_frames=None, saveas='video', retries=10, skip_method=None, skip_max=None,):
        self.folder = folder
        self.saveas = saveas
        self.n_frames = n_frames
        self.skip_method = skip_method
        self.skip_max = skip_max
        self.retries = retries
        if skip_method is not None:
            self.is_frame_invalid = skip_methods[skip_method]
        self.video = Video(shape=(854, 480))
        
    def capture_frame(self, cap, url):
        success, frame = cap.read(); r = 0
        while not success and r < self.retries:
            cap = cv2.VideoCapture(url)
            success, frame = cap.read(); r += 1
        return cap, frame # frame is None if retries are exceeded

    def capture(self, code, n_frames=None, path=None):
        if n_frames is None: n_frames = self.n_frames
        if path is not None: writer = self.video.writer(path)
        url = self.url.format(code)
        cap = cv2.VideoCapture(url)
        skipped, appended, frames = 0, 0, []
        while(True):
            cap, frame = self.capture_frame(cap, url)
            if frame is None:
                print(f'CONNECTION LOST. CODE {code}. MAX RETRIES ({self.retries}) EXCEEDED. AT {now("%Y-%m-%d %X")}.')
                break
            if self.skip_method is not None:
                if self.is_frame_invalid(frame):
                    skipped +=1
                    if self.skip_max is not None:
                        if skipped == self.skip_max:
                            print(f'CAPTURE STOPPED. CODE {code}. MAX SKIPPED FRAMES ({self.skip_max}) EXCEEDED. AT {now("%Y-%m-%d %X")}.')
                            break
                    continue
            filename = f'CODE{code} {now("%Y-%m-%d %H-%M-%S-%f")[:-5]}'  # [:-5] keeps only the first microsecond decimal
            if path is None:
                frames.append([filename, frame])
            else:
                writer.write(frame)
            appended += 1; frame = None
            if n_frames is not None:
                if appended >= n_frames: break
        cv2.destroyAllWindows(); cap.release()
        if path is None:
            return frames
        else:
            writer.release()
            return appended
    
    def cap_report(status, code, stamp):
        if not status:
            print(f'FAILED TO RECORD. CODE {code}, AT {stamp}.')
            return {'message': 'FAILED', 'code': code, 'start': stamp, 'end': now()}

    def record(self, code, folder=None, n_frames=None, saveas=None):
        stamp = now()
        if n_frames is None: n_frames = self.n_frames
        if folder is None: folder = self.folder
        if saveas is None: saveas = self.saveas
        path = f'{folder}/{code}'
        blob_name = f'{path}/CODE{code} {stamp}.mp4'
        if saveas == 'image':
            frames = self.capture(code, n_frames)
            if not len(frames): return Recorder.cap_report(False, code, stamp)
            upload_frames(frames, f'{path}/{stamp}')
        elif saveas == 'video':
            with NamedTemporaryFile() as temp:
                tname = f"{temp.name}.mp4"
                n_cap = self.capture(code, n_frames, tname)
                if not n_cap: return Recorder.cap_report(False, code, stamp)
                gcs.upload_from_filename(tname, blob_name, bucket_name, 'video/mp4')
        elif saveas == 'video-from-list':
            frames = self.capture(code, n_frames)
            if not len(frames): return Recorder.cap_report(False, code, stamp)
            frames_arr = [frame for _, frame in frames]
            upload_video(frames_arr, blob_name)
        end = now()
        if saveas != 'video': n_cap = len(frames)
        print(f'UPLOAD SUCCESS. CODE {code}, FRAMES {n_cap}, STARTED AT {stamp}, FINISHED AT {end}')
        return {'message': 'SUCCESS', 'code': code, 'frames': n_cap, 'start': stamp, 'end': end}

    def report_finished(self, stamp):
        print(f"UPLOAD MANY FINISHED. STARTED AT {stamp}, FINISHED AT {now()}.")