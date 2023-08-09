import os, pandas as pd, numpy as np
from IPython.display import clear_output as co 

def files_tree_sizes(path='.', report_level=5, level=0, state=[]):
    sizes = pd.Series()
    try: n_entries = len(os.listdir(path))
    except: n_entries = None    
    state = state.copy()
    state.append([str(0), str(n_entries)])
    with os.scandir(path) as it:
        for i, entry in enumerate(it):
            state[level][0] = str(i+1)
            if level <= report_level:
                co(wait=True)
                formatted_state = ' - '.join(map('/'.join, state))
                print(f'Scanning items: {formatted_state}')
            try:
                if entry.is_file():
                    sizes[entry.path] = entry.stat().st_size
                elif entry.is_dir():
                    sizes = pd.concat([sizes, get_files_sizes(entry.path, report_level=report_level, level=level+1, state=state)], 0)
            except:
                sizes[entry.path] = np.nan
    return sizes

def remove_files(file_list):
    fail = []
    for i, file in enumerate(file_list):
        try:
            if os.path.exists(file):
                os.remove(file)
        except:
            fail.append(file)
        co(wait=True); print(f'files removed: {i}/{len(file_list)}')
    return fail