import pandas as pd, requests

class cameras:

    api_root = 'https://bolsao-api-j2fefywbia-rj.a.run.app'

    def __init__(self, src, monitor, logic='any', merge=False):
        self.src = src
        self.monitor = monitor
        self.logic = logic
        self.merge = merge
        self.static = self.read_df('cameras')
        self.update()

    def read_df(self, src):
        return pd.DataFrame(requests.get(f'{self.api_root}/{src}').json())

    def in_alert_msk(self):
        if self.logic == 'any':
            return self.state[self.monitor].any(axis=1)
        elif self.logic == 'all':
            return self.state[self.monitor].all(axis=1)

    def update(self, report=False):
        self.state = self.read_df(self.src)
        if self.merge: self.state =  pd.merge(self.static, self.state, how='inner', on='cluster_id')
        self.in_alert = self.state['Codigo'][self.in_alert_msk()].tolist()
        self.normal = list(set(self.state['Codigo']).difference(self.in_alert))
        if report: self.report()
        
    def report(self): display(pd.DataFrame(
        [[len(self.in_alert), len(self.normal), len(self.state)]],
        columns=['In alert', 'Normality', 'Total'], index=[self.src]
    ))

from datetime import datetime as dt
import pytz; tzbr = pytz.timezone('Brazil/East')
from google.oauth2 import service_account
from google.cloud import storage
import cv2, multiprocessing
from tempfile import NamedTemporaryFile

def now(fmt="%Y-%m-%d %H-%M-%S"):
    return dt.now(tzbr).strftime(fmt)

# google cloud storage
bucket_name = 'city-camera-images'

def get_bucket():
    credentials = service_account.Credentials.from_service_account_file('../../../../Apps/Python/bolsao-api/credentials/pluvia-360323-35cd376d5958.json')
    return storage.Client(credentials=credentials).get_bucket(bucket_name)

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
    
    def download(self, blob, path):
        return get_bucket().blob(blob).download_to_filename(path)   
    
    def upload_video(self, frames, blob):
        bucket = get_bucket()
        with NamedTemporaryFile() as temp:
            tname = f"{temp.name}.mp4"
            write_video(frames, tname, fps=3, codec='mp4v')
            bucket.blob(blob).upload_from_filename(tname, content_type="video/mp4")

    def upload_frames(self, frames, path):
        bucket = get_bucket()
        with NamedTemporaryFile() as temp:
            tname = f"{temp.name}.jpg"
            for j, (filename, frame) in enumerate(frames):
                blob = f'{path}/{filename}.jpg'
                cv2.imwrite(tname, frame)
                bucket.blob(blob).upload_from_filename(tname, content_type="image/jpeg")

    def record(self, code):
        path = f'{self.folder}/{code}'
        stamp = now()
        frames = self.capture(code)
        if self.saveas == 'image':
            self.upload_frames(frames, f'{path}/{stamp}')
        elif self.saveas == 'video':
            frames_arr = [frame for _, frame in frames]
            blob = f'{path}/CODE{code} {stamp}.mp4'
            self.upload_video(frames_arr, blob)

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