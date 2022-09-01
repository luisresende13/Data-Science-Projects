import os, pandas as pd

def read_csv_folder(path, index_col=0):
    return (pd.read_csv(path+file, index_col=index_col) for file in os.listdir(path) if file.endswith('.csv'))

class data:
    
    def __init__(self, path=None, load=True, exts=['.csv', '.xlsx'], include=None):
        self.path = path
        self.exts = exts
        self.files = os.listdir(self.path)
        self.dfs, self.failed = {}, {}
        if load:
            self.load(reload=True, include=include)

            
    def load(self, path=None, reload=False, include=None):
        
        if path is not None:
            self.path = path
            self.files = os.listdir(self.path)
        
        if reload:
            self.dfs, self.failed = {}, {}
        
        cur_dfs = list(self.dfs.keys())

        for i, file in enumerate(self.files):
            name = file.split('.')[0]
            co(wait=True); print(f'{i+1}/{len(self.files)} - {name}')
            if name in cur_dfs:
                continue
            if include is not None:
                if name not in include:
                    continue                    
            for ext in self.exts:                    
                if file.endswith(ext):
                    try: self.dfs[name] = pd.read_csv(self.path+file)
                    except Exception: self.failed[name] = {'error': Exception}