import cv2 as cv
from google.cloud import storage
import os
from tempfile import NamedTemporaryFile

class GCS:
    '''
    TO DO
    correct bug in 'download_folder_from_bucket' in which an extra empty folder is created
    '''
    def __init__(self, service_account_json=None) -> None:
        if service_account_json is not None:
            self.storage_client = storage.Client.from_service_account_json(service_account_json)
        else:
            self.storage_client = storage.Client()

    def get_bucket(self, bucket_name:str):
        return self.storage_client.get_bucket(bucket_name)
    
    def list_buckets(self):
        return [bucket.name for bucket in self.storage_client.list_buckets()]
    
    def is_bucket_in_storage(self, bucket_name:str) -> bool:
        return bucket_name in self.list_buckets() # CHANGED
    
    def get_blob(self, blob_name:str, bucket_name:str):
        return self.storage_client.get_bucket(bucket_name).blob(blob_name)

    def list_blobs(self, bucket_name:str, folder:str=""):
        return [blob.name for blob in self.get_bucket(bucket_name).list_blobs(prefix=folder)]
    
    def is_blob_in_bucket(self, blob_name:str, bucket_name:str) -> bool:
        return blob_name in self.list_blobs(bucket_name) # CHANGED
    
    def get_folder(self, bucket_name:str, folder_path:str):
        return self.storage_client.get_bucket(bucket_name).list_blobs(prefix=folder_path)

    def upload_from_filename(self, filename:str, blob_name:str, bucket_name:str, content_type:str="image/jpeg"): # CHANGED
        self.get_blob(blob_name, bucket_name).upload_from_filename(filename, content_type)

    def upload_from_file(self, file, blob_name:str, bucket_name:str, content_type:str="image/jpeg"): # CHANGED
        with NamedTemporaryFile() as temp:
            tname = "".join([str(temp.name),".jpg"])
            cv.imwrite(tname, file)
            self.upload_from_filename(tname, blob_name, bucket_name, content_type)

    def download_to_filename(self, filename, blob_name, bucket_name): # NEW
        return self.get_blob(blob_name, bucket_name).download_to_filename(filename)

    def download_from_bucket(self, blob_name:str, download_path:str, bucket_name:str, in_memory:bool=True):
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

    def download_folder_from_bucket(self, folder_path:str, bucket_name:str, download_path:str="download"):
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

        blobs = self.get_folder(bucket_name, folder_path)
        
        for blob in blobs:
            output_path = f'{download_path}{blob.name.replace(":","-")}'
            if output_path.endswith("/"): # CHANGED
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                self.download_folder_from_bucket(output_path, bucket_name)
            else:
                blob.download_to_filename(output_path)