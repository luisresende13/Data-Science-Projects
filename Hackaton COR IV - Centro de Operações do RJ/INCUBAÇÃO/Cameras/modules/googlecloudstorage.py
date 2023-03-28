import cv2 as cv, numpy as np
from google.cloud import storage
import os
from tempfile import NamedTemporaryFile
from IPython.display import clear_output as co
from time import time

class GCS:
    '''
    TO DO
    correct bug in 'download_folder_from_bucket' in which an extra empty folder is created
    '''
    def __init__(self, service_account_json:str=None, user_project:str=None, bucket_name:str=None) -> None:
        self.storage_client = storage.Client.from_service_account_json(service_account_json) if service_account_json is not None else storage.Client()
        self.user_project = user_project
        self.bucket_name = bucket_name
        self.bucket = self.get_bucket(bucket_name) if bucket_name is not None and type(bucket_name) is not float else None

    def get_bucket(self, bucket_name:str=None):
        if bucket_name is None or type(bucket_name) is float: return self.bucket
        return self.storage_client.bucket(bucket_name, user_project=self.user_project)
    
    def list_buckets(self):
        return [bucket.name for bucket in self.storage_client.list_buckets()]
    
    def is_bucket_in_storage(self, bucket_name:str=None) -> bool:
        if bucket_name in [None, np.nan]: bucket_name = self.bucket_name
        return bucket_name in self.list_buckets() # CHANGED
    
    def get_blob(self, blob_name:str, bucket_name:str=None):
        return self.get_bucket(bucket_name).blob(blob_name)

    def list_blobs(self, prefix:str, delimiter:str=None, bucket_name:str=None):
        return self.get_bucket(bucket_name).list_blobs(prefix=prefix, delimiter=delimiter)
    
    def is_blob_in_bucket(self, blob_name:str, bucket_name:str=None) -> bool:
        return self.get_blob(blob_name, bucket_name).exists()
    
    def get_folder(self, folder:str, bucket_name:str=None): # REMOVE. EQUAL TO LIST_BLOBS
        return self.get_bucket(bucket_name).list_blobs(prefix=folder)

    def upload_from_filename(self, filename:str, blob_name:str, bucket_name:str=None, content_type:str="image/jpeg", overwrite:bool=False): # CHANGED
        if not overwrite and self.is_blob_in_bucket(blob_name, bucket_name):
            print(f'UPLOAD FAILED. BLOB ALREADY EXISTS. FILE: {filename} · BLOB: {blob_name} · BUCKET: {bucket_name}')
            return False
        try:
            blob = self.get_blob(blob_name, bucket_name)
            blob.upload_from_filename(filename, content_type)
            return True
        except: return False

    def upload_from_file(self, file, blob_name:str, bucket_name:str=None, content_type:str="image/jpeg"): # CHANGED
        with NamedTemporaryFile() as temp:
            tname = "".join([str(temp.name),".jpg"])
            cv.imwrite(tname, file)
            self.upload_from_filename(tname, blob_name, bucket_name, content_type)

    def download_to_filename(self, filename:str, blob_name:str, bucket_name:str=None): # NEW
        return self.get_blob(blob_name, bucket_name).download_to_filename(filename)

    def download_to_folder(
        self, folder:str, prefix:str, delimiter:str, bucket_name:str=None,
        overwrite:bool=False, report_freq=50, skip=0
    ): # NEW
        if not folder.endswith('/'):  folder += '/'
        blobs = [blob.name for blob in self.list_blobs(prefix, delimiter, bucket_name) if not blob.name.endswith('/')]
        start = time()
        n = len(blobs) - skip
        success = 0
        for i, blob_name in enumerate(blobs):
            if i >= skip:
                report = (i+1) % report_freq == 0
                if report:
                    j = i - skip
                    running = time() - start
                    running_min = round(running / 60, 1)
                    download_rate = round(running / j, 4)
                    finish_time_estimate = download_rate * (n - j)
                    finish_time_estimate_min = round(finish_time_estimate / 60, 1)
                    co(True); print(); print(f'PREFIX: {prefix} · RUNNING: {running_min} min · RATE: {download_rate} s / file · FINISH-ESTIMATE: {finish_time_estimate_min} min · PROGRESS: {i+1}/{len(blobs)} · DOWNLOADS: {success+1}/{n}')
                filepath = f'{folder}{blob_name}'.replace(':', '-')
                path = '/'.join(filepath.split('/')[:-1])
                if not overwrite and os.path.exists(filepath):
                    print(); print(f'DOWNLOAD FAILED. FILE ALREADY EXISTS. FILE: {filepath} · BLOB: {blob_name} · BUCKET: {bucket_name} · ({i}/{len(blobs)})')
                    continue
                if not os.path.exists(path): os.makedirs(path)
                self.download_to_filename(filepath, blob_name, bucket_name); success += 1
        
    def download_from_bucket(self, blob_name:str, download_path:str, bucket_name:str=None, in_memory:bool=True):
        '''
        params - 
            blob_name: image filename, with format included
            download_path: path to download folder
            bucket_name: bucket to be downloaded from
            service_account_json: path to json file with Google Cloud credentials to access bucket 'bucket_name'
        '''
        blob = self.get_blob(blob_name, bucket_name)
        
        if not download_path.endswith("/"): # CHANGED
            download_path += "/"
            
        if not os.path.exists(download_path):
            os.makedirs(download_path)

        output_path = f'{download_path}{blob.name.replace(":","-")}'
        blob.download_to_filename(output_path)
        
        if in_memory:
            return cv.imread(output_path)

    def download_folder_from_bucket(self, folder_path:str, bucket_name:str=None, download_path:str="download"):
        '''
        
        '''
        # https://stackoverflow.com/questions/49748910/python-download-entire-directory-from-google-cloud-storage
        
        if download_path is not None:
            if download_path[len(download_path)-1] != "/":
                download_path += "/"
            if not os.path.exists(download_path):
                os.makedirs(download_path)
        
        if folder_path[len(folder_path)-1] != "/":
            folder_path += "/"

        if not os.path.exists(download_path+"/"+folder_path):
            os.makedirs(download_path+"/"+folder_path)

        blobs = self.get_folder(folder_path, bucket_name)
        
        for blob in blobs:
            output_path = f'{download_path}{blob.name.replace(":","-")}'
            if output_path.endswith("/"): # CHANGED
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                self.download_folder_from_bucket(output_path, bucket_name)
            else:
                blob.download_to_filename(output_path)