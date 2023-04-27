import os, cv2, pandas as pd
from datetime import datetime as dt
from IPython.display import clear_output as co

def inner_subdirs(folder, ext='.mp4'):
    if folder.endswith('/'): folder = folder[:-1]
    folder_depth = len(folder.split('/'))
    in_subdirs = []
    for path, subdirs, files in os.walk(folder):
        if os.path.isdir(path) and any([file.endswith(ext) for file in files]):
            subpath = '/'.join(path.split('\\')[1:])
            in_subdirs.append(subpath)
    return in_subdirs

class VideoAnnotator:

    def add_timestamp_to_frame(frame, text):
        cv2.putText(frame, text, org=(542, 27), fontFace=cv2.FONT_HERSHEY_COMPLEX,
        fontScale=0.67, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_8)
        cv2.putText(frame, text, org=(540, 25), fontFace=cv2.FONT_HERSHEY_COMPLEX,
        fontScale=0.67, color=(40, 230, 230), thickness=2, lineType=cv2.LINE_8)
        return frame

    def __init__(self, fps:int=3, shape:tuple=(854, 480), codec:str='mp4v'):
        self.fps = fps; self.shape = shape; self.codec = codec

    def write_timestamp_to_videos(self, folder, path, to_folder, overwrite=True):
        """
        folder: folder relative to current absolute path containing `path` to video file
        path: path relative to `folder` to video file containing a timestamp in file name in the format: `XXXX %Y-%m-%d %H-%M-%S.mp4`
        """
        file_name = path.split('/')[-1]
        stamp = ' '.join(file_name.split(' ')[1:])
        timestamp = dt.strptime(stamp, '%Y-%m-%d %H-%M-%S.mp4')
        offset = pd.offsets.Second() / self.fps
        full_path = f'{folder}/{path}'
        to_full_path = f'{to_folder}/{path}'
        full_folder = '/'.join(to_full_path.split('/')[:-1])
        if not overwrite and os.path.exists(to_full_path):
            print(f'ANNOTATE VIDEO TIMESTAMP FAILED. FILE ALREADY EXISTS · FILE: {to_full_path}')
            return False
        if not os.path.exists(full_folder): os.makedirs(full_folder)
        cap = cv2.VideoCapture(full_path)
        video = cv2.VideoWriter(to_full_path, cv2.VideoWriter_fourcc(*self.codec), self.fps, self.shape)
        while cap.isOpened():
            r, frame = cap.read()
            if not r:
                break
            frame = VideoAnnotator.add_timestamp_to_frame(frame, timestamp.strftime('OCTA %d/%m/%Y %H:%M:%S'))
            timestamp += offset
            video.write(frame)
        cap.release(); video.release(); cv2.destroyAllWindows()
        return True
    
    def write_timestamp_to_nested_videos(self, folder, to_folder, ext='.mp4', overwrite=True, report_freq=10):
        "Annotates timestamps in all videos found matching the `ext` extesion inside all nested subdirectories inside `folder`"
        if folder.endswith('/'): folder = folder[:-1]
        folder_depth = len(folder.split('/'))
        paths = []
        for path, subdirs, files in os.walk(folder):
            for name in files:
                if name.endswith(ext):
                    full_path = os.path.join(path, name).replace('\\', '/')
                    # file path relative to `folder` ending with `ext` ...
                    subpath = '/'.join(full_path.split('/')[folder_depth:])
                    paths.append(subpath)
        success = 0
        n = len(paths)
        for i, subpath in enumerate(paths):
            if self.write_timestamp_to_videos(folder, subpath, to_folder, overwrite): success += 1
            if report_freq is not None and (i + 1) % report_freq == 0:
                co(True); print(f'VIDEO TIMESTAMP ANNOTATION · DONE: {i + 1}/{n} · SUCCESS: {success}/{n}')

    def concatenate_videos_from_folder(self, folder, path, ext:str='.mp4', overwrite=True):
        """
        prameters:
            folder - folder containing video files matching extension `ext` to be concatenated
            path - path to save video file generated after concatenation
            overwrite - whether or not to overwrite file in `path`
        """
        if folder.endswith('/'): folder = folder[:-1]        
        to_full_path = f'{folder}/{path}'
        to_full_folder = '/'.join(to_full_path.split('/')[:-1])
        if not overwrite and os.path.exists(path):
            print(f'CONCAT VIDEOS FROM FOLDER FAILED · FILE ALREADY EXISTS · FILE: {path}')
            return False
        if not os.path.exists(to_full_folder): os.makedirs(to_full_folder)
        video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*self.codec), self.fps, self.shape)
        for file_name in sorted(os.listdir(folder)):
            if file_name.endswith(ext):
                stamp = ' '.join(file_name.split(' ')[1:])
                timestamp = dt.strptime(stamp, f'%Y-%m-%d %H-%M-%S{ext}')
                file_path = f'{folder}/{file_name}'
                cap = cv2.VideoCapture(file_path)
                while cap.isOpened():
                    r, frame = cap.read()
                    if not r:
                        break
                    video.write(frame)
                cap.release()
        video.release(); cv2.destroyAllWindows()
        return True
        
    def concatenate_videos_by_date_from_nested_folders(
        self, base_folder:str, to_base_folder:str,
        ext:str='.mp4', overwrite:bool=True, report_freq=10
    ):
        folders = inner_subdirs(base_folder, ext)
        success = 0
        n = len(folders)
        for i, folder in enumerate(folders):
            if self.concatenate_videos_by_date_from_folder(folder, base_folder, to_base_folder, ext, overwrite):
                success += 1
            if report_freq is not None and (i + 1) % report_freq == 0:
                co(True); print(f'CONCAT VIDEOS BY DATE FROM NESTED FOLDERS · DONE: {i + 1}/{n} · FOLDER: {folder}')
        
    def concatenate_videos_by_date_from_folder(
        self, folder:str, base_folder:str, to_base_folder:str,
        ext:str='.mp4', overwrite:bool=True
    ):
        """
        """
        for path in [folder, base_folder, to_base_folder]:
            if path.endswith('/'): path = path[:-1]
        full_folder = f'{base_folder}/{folder}'
        to_full_folder = f'{to_base_folder}/{folder}'
        files_info = []
        for file_name in os.listdir(full_folder):
            if file_name.endswith(ext):
                file_path = f'{full_folder}/{file_name}'
                file_name_split = file_name.split(' ')
                code = file_name_split[0]
                stamp = ' '.join(file_name_split[1:])
                timestamp = dt.strptime(stamp, '%Y-%m-%d %H-%M-%S.mp4')
                files_info.append([timestamp, code, file_path])
        files_info = pd.DataFrame(files_info, columns=['timestamp', 'code', 'file_path']).set_index('timestamp', drop=True)
        files_info['date'] = files_info.index.date
        code = files_info.iloc[0]['code']; files_info.drop('code', axis=1, inplace=True)
        for date in files_info['date'].unique():
            to_file_name = date.strftime(f'{code} %Y-%m-%d{ext}')
            to_full_path = f'{to_full_folder}/{to_file_name}'
            if not overwrite and os.path.exists(to_full_path):
                print(f'CONCAT VIDEOS BY DATE FROM FOLDER (FAILED) · FILE ALREADY EXISTS · FILE: {to_file_name}')
                continue
            if not os.path.exists(to_full_folder): os.makedirs(to_full_folder)
            video = cv2.VideoWriter(to_full_path, cv2.VideoWriter_fourcc(*self.codec), self.fps, self.shape)
            files = files_info[files_info['date']==date] # datetime index filtering
            for file_path in files['file_path'].sort_index():
                cap = cv2.VideoCapture(file_path)
                while cap.isOpened():
                    r, frame = cap.read()
                    if not r:
                        break
                    video.write(frame)
                cap.release()
            video.release(); cv2.destroyAllWindows()
            print('CONCAT VIDEOS BY DATE FROM FOLDER (SUCCESS) · FILE-CREATED: ', to_file_name)
        return True