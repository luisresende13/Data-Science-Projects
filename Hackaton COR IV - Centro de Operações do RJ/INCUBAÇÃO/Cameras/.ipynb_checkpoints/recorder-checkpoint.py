from datetime import datetime as dt
import pytz; tzbr = pytz.timezone('Brazil/East')
from googlecloudstorage import GCS
from tempfile import NamedTemporaryFile
import cv2, multiprocessing

def now(fmt="%Y-%m-%d %H-%M-%S"):
    return dt.now(tzbr).strftime(fmt)

# google cloud storage settings

gcs = GCS('../../../../Apps/Python/bolsao-api/credentials/pluvia-360323-35cd376d5958.json') # YOUR GOOGLE CLOUD SERVICE ACCOUNT JSON FILE PATH
bucket_name = 'city-camera-images' # YOUR GOOGLE CLOUD STORAGE BUCKET NAME

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

    def __init__(self, folder='./', saveas='video', n_frames=None, skip_first=0, retries=10):
        self.folder = folder
        self.saveas = saveas
        self.n_frames = n_frames
        self.skip_first = skip_first
        self.retries = retries
        
    def initialize_capture(self, code):
        skipped, retries = 0, 0
        cap = cv2.VideoCapture(self.url.format(code))
        while skipped < self.skip_first and retries < self.retries:
            success, frame = cap.read()
            while not success and retries < self.retries:
                skipped, cap = 0, cv2.VideoCapture(self.url.format(code))
                success, frame = cap.read(); retries += 1
                print(f'Code: {code}, Skipped: {skipped}, Retries: {retries}')
            if success: skipped += 1
        return retries < self.retries, cap
    
    def capture(self, code):
        online, cap = self.initialize_capture(code)
        if not online: return []
        i, frames = 0, []
        while(True):
            success, frame = cap.read()
            while not success:
                online, cap = self.initialize_capture(code)
                if not online: return []
                success, frame = cap.read()
            filename = f'CODE{code} {now(fmt="%Y-%m-%d %H-%M-%S-%f")[:-4]}'            
            frames.append([filename, frame]); i += 1
            if self.n_frames is not None:
                if i >= self.n_frames: break
        return frames        
    
    def record(self, code):
        path = f'{self.folder}/{code}'
        stamp = now()
        frames = self.capture(code)
        if self.saveas == 'image':
            upload_frames(frames, f'{path}/{stamp}')
        elif self.saveas == 'video':
            frames_arr = [frame for _, frame in frames]
            blob = f'{path}/CODE{code} {stamp}.mp4'
            upload_video(frames_arr, blob)

    def record_many(self, codes, workers='auto'):
        if not len(codes): print('EMPTY CODE LIST PROVIDED. TASK SKIPPED.'); return
        if workers == 'auto': workers = len(codes)
        pool = multiprocessing.Pool(processes=workers)
        codes = [(code,) for code in codes]
        stamp = now(); pool.starmap(self.record, codes)
        self.report_finished(stamp)
        return

    def report_finished(self, stamp):
        print(f"Job started at'{stamp}' finished at '{now()}'.")