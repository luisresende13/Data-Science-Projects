import cv2, numpy as np, pandas as pd
from datetime import datetime as dt
from IPython.display import clear_output as co

# default `filename_to_timestamp_function`
def timestamp_parser(path):
    file_name = path.split('/')[-1]
    stamp = ' '.join(file_name.split(' ')[1:])
    return dt.strptime(stamp, '%Y-%m-%d %H-%M-%S.mp4')

class VideoProcessor:
        
    def __init__(self, fps=3, frame_dimension=1, frame_processing_function=None, filename_to_timestamp_function=timestamp_parser):
        """  """
        self.fps = fps
        self.frame_dimension = frame_dimension
        self.frame_processing_function = frame_processing_function
        self.filename_to_timestamp_function = filename_to_timestamp_function
        
    def process_labeled_videos(self, videos_dataframe, path_key, print_each=None):
        """
        parameters:
            videos - pandas dataframe containing a column with name `path_key`, with paths to a set of videos.
            path_key - name of column containing path to videos in local file system
        """
        i, n  = 0, len(videos_dataframe)
        results = []
        for idx, video_object in videos_dataframe.iterrows():
            frames_results = self.process_frames(video_object, path_key)
            results += frames_results
            i += 1
            if print_each is not None and i % print_each == 0:
                co(True); print(f'LABELED VIDEOS PROCESSING · PROGRESS: {i}/{n}')
        return results

    def process_frames(self, video_object, path_key):
        path = video_object[path_key]
        timestamp = None
        if self.filename_to_timestamp_function is not None:
            timestamp = self.filename_to_timestamp_function(path)
            offset = pd.offsets.Second() / self.fps
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"CANNOT OPEN VIDEO CAPTURE · PATH: {path}")
            return []
        frames_results = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break # stream finished
            if self.frame_dimension == 1: # 1D flat frame
                frame = np.reshape(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), -1)
            if self.frame_dimension == 2: # 2D gray scale frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.filename_to_timestamp_function is not None:
                timestamp += offset # frame time stamp update
            result = self.frame_processing_function(frame, metadata=video_object.to_dict(), timestamp=timestamp)
            frames_results.append(result)
        cap.release(); cv2.destroyAllWindows()
        return frames_results