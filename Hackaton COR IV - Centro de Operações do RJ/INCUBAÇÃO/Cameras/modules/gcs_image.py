import os, json, numpy as np, cv2
from tempfile import NamedTemporaryFile

class Video:
    
    def load_image_from_bytes(bytes_string):
        return cv2.imdecode(np.frombuffer(bytes_string ,dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    def __init__(self, shape:tuple=None, fps:int=3, ext:str='.mp4', codec:str='mp4v', skipper=None, gcs=None):
        self.shape = shape
        self.fps = fps
        self.ext = ext
        self.codec = codec
        self.skipper = skipper
        self.gcs = gcs

    def read_images_from_folder(self, folder:str, ext:str='.jpg', writer=None, report=True):
        images, fail = [], []
        paths = os.listdir(folder)
        for path in paths:
            if path.endswith(ext):
                image_path = f'{folder}/{path}'
                image = cv2.imread(image_path)
                if image is None:
                    if report:
                        print(f'\n- IMAGE READ FAILED. PATH: {image_path}')
                        fail.append({'TYPE': 'IMAGE READ/WRITE FAILED', 'MESSAGE': 'IMAGE IS NONE', 'OK': False, 'PATH': path, 'FOLDER': folder})
                    continue
                if self.skipper is not None and self.skipper(image):
                    if report:
                        print(f'\n- IMAGE READ/WRITE SKIPPED. PATH: {image_path}')
                        fail.append({'TYPE': 'IMAGE READ/WRITE FAILED', 'MESSAGE': 'IMAGE READ/WRITE SKIPPED', 'OK': False, 'PATH': path, 'FOLDER': folder})
                    continue
                if writer is not None: writer.write(image)
                else: images.append(image)
        if report and len(fail): print(f'\n- IMAGES READ/WRITE FAILED · FAIL: {len(fail)} · FOLDER: {folder}')
        if writer is not None: return len(images), writer
        return len(images), images

    def read_images_from_bucket(self, bucket_name:str, prefix:str, delimiter:str=None, ext:str='.jpg', writer=None, report:bool=False):
        images, fail = [], []
        for blob in self.gcs.list_blobs(prefix, delimiter, bucket_name):
            if blob.name.endswith(ext):
                image = Video.load_image_from_bytes(blob.download_as_string())
                if image is None:
                    if report:
                        print(f'\n- IMAGE READ FAILED. BLOB: {blob.name}')
                        fail.append({'TYPE': 'IMAGE READ/WRITE FAILED', 'MESSAGE': 'IMAGE IS NONE', 'OK': False, 'BLOB': blob.name})
                    continue
                if self.skipper is not None and self.skipper(image):
                    if report:
                        print(f'\n- IMAGE DOWNLOAD/WRITE SKIPPED. PREFIX: {prefix} · BUCKET: {bucket_name}')
                        fail.append({'TYPE': 'IMAGE DOWNLOAD/WRITE FAILED', 'MESSAGE': 'IMAGE DOWNLOAD/WRITE SKIPPED', 'OK': False, 'PREFIX': prefix, 'BUCKET': bucket_name})
                    continue
                if writer is not None: writer.write(image); images.append(1)
                else: images.append(image)
        if report and len(fail): print(F'\n- IMAGES DOWNLOAD/WRITE FAILED · IMAGES-FAILED: {len(fail)} · PREFIX: {prefix} · BUCKET: {bucket_name}')
        if writer is not None: return len(images), writer
        return len(images), images

    def write(self, images:list, path:str):
        if self.shape is None:
            try: height, width, _ = images[0].shape; shape = (width, height)
            except:
                print('\nVIDEO WRITE FAILED. SHAPE COULD NOT BE DETERMINED AUTOMATICALLY.')
                return False
        else: shape = self.shape
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(path, fourcc, self.fps, shape)
        for frame in images: writer.write(frame)
        cv2.destroyAllWindows(); writer.release()
        return len(images)

    def write_from_folder_of_images(self, path:str, folder:str, ext:str='.jpg', report:bool=True):
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(path, fourcc, self.fps, self.shape)
        n_images, writer = self.read_images_from_folder(folder, ext, writer, report)
        cv2.destroyAllWindows(); writer.release()
        return n_images

    def write_from_bucket_folder_of_images(self, path:str, bucket_name:str, folder:str, ext:str='.jpg', report:bool=False):
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(path, fourcc, self.fps, self.shape)
        n_images, writer = self.read_images_from_bucket(bucket_name, folder, None, ext, writer, report)
        cv2.destroyAllWindows(); writer.release()
        return n_images

    def upload(self, filename:str, blob_name:str, bucket_name:str, content_type:str='video/mp4', overwrite:bool=False):
        return self.gcs.upload_from_filename(filename, blob_name, bucket_name, content_type, overwrite)

    def upload_from_images(self, images:list, blob_name:str, bucket_name:str, content_type:str='video/mp4', overwrite:bool=False):
        if not len(images):
            print(f'\n* VIDEO UPLOAD FAILED. EMPTY FRAME LIST PROVIDED. BLOB: {blob_name} · BUCKET: {bucket_name}')
            return False
        if not overwrite and self.gcs.is_blob_in_bucket(blob_name, bucket_name):
            print(f'\n* VIDEO UPLOAD FAILED. BLOB ALREADY EXISTS. BLOB: {blob_name} · BUCKET: {bucket_name}')
            return False
        with NamedTemporaryFile() as temp:
            tname = f"{temp.name}{self.ext}"
            write = self.write(images, tname)
            upload = self.upload(tname, blob_name, bucket_name, content_type, overwrite=True) # overwrite check already done
            if upload: print(f'\n* VIDEO UPLOAD SUCCESS. IMAGES: {write} · BLOB: {blob_name} · BUCKET: {bucket_name}')            
            return upload

    def upload_from_loaded_folder_of_images(self, folder:str, blob_name:str, bucket_name:str, ext:str='.jpg', content_type:str='video/mp4', overwrite:bool=False):
        if not overwrite and self.gcs.is_blob_in_bucket(blob_name, bucket_name):
            print(f'VIDEO UPLOAD FAILED. BLOB ALREADY EXISTS. BLOB: {blob_name} · BUCKET: {bucket_name}')
            return False
        return self.upload_from_images(self.read_images_from_folder(folder, ext)[1], blob_name, bucket_name, content_type, overwrite=True) # overwrite check already done

    def upload_from_folder_of_images(self, folder:str, blob_name:str, bucket_name:str, ext:str='.jpg', content_type:str='video/mp4', overwrite:bool=False, report:bool=True):
        if not overwrite and self.gcs.is_blob_in_bucket(blob_name, bucket_name):
            print(f'\n* VIDEO UPLOAD FAILED. BLOB ALREADY EXISTS. BLOB: {blob_name} · BUCKET: {bucket_name}')
            return False
        with NamedTemporaryFile() as temp:
            tname = f"{temp.name}{self.ext}"
            write = self.write_from_folder_of_images(tname, folder, ext, report)
            if not write:
                print(f'\n* VIDEO UPLOAD FAILED. NO BLOBS FOUND · FOLDER: {folder} · MATCHING EXTENSION: {ext} · IMAGES: {write} · FOLDER: {folder}')
                return False
            upload = self.upload(tname, blob_name, bucket_name, content_type, overwrite=True) # overwrite check already done
            if upload: print(f'\n* VIDEO UPLOAD SUCCESS · IMAGES: {write} BLOB: {to_blob_name} · BUCKET: {to_bucket_name}')
            return upload
        
    def upload_from_bucket_folder_of_images(
        self, folder:str, from_bucket_name:str,
        to_blob_name:str, to_bucket_name:str, ext:str='.jpg',
        content_type:str='video/mp4', overwrite:bool=False,
        report:bool=False
    ):
        if not overwrite and self.gcs.is_blob_in_bucket(to_blob_name, to_bucket_name):
            print(f'\n* VIDEO UPLOAD FAILED. BLOB ALREADY EXISTS · BLOB: {to_blob_name} · BUCKET: {to_bucket_name}')
            return False
        with NamedTemporaryFile() as temp:
            tname = f"{temp.name}{self.ext}"
            write = self.write_from_bucket_folder_of_images(tname, from_bucket_name, folder, ext, report)
            if not write:
                print(f'\n* VIDEO UPLOAD FAILED. NO BLOBS FOUND · FOLDER: {folder} · MATCHING EXTENSION: {ext} · IMAGES: {write} · BUCKET: {from_bucket_name}')
                return False
            upload = self.upload(tname, to_blob_name, to_bucket_name, content_type, overwrite=True) # overwrite check already done
            if upload: print(f'\n* VIDEO UPLOAD SUCCESS · IMAGES: {write} · BLOB: {to_blob_name} · BUCKET: {to_bucket_name}')
            return upload
    