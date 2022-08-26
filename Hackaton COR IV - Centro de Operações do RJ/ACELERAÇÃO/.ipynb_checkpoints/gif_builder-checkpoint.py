import os, imageio as iio, numpy as np
from pathlib import Path
# from pygifsicle import optimize
from IPython.display import clear_output as co

def folder_to_gif(folder, gif_path):
    frames = []
    file_iter = Path(folder).iterdir()
    n_files = len(os.listdir(folder))
    for file in file_iter:
        co(wait=True); print(f'{i+1}/{n_files}')
        if file.is_file():
            frames.append(iio.imread(file))
    frames = np.stack(
        frames,
        axis=0
    )
    iio.imwrite(gif_path, frames, mode="I")
#     optimize(gif_path)
    print(f'Done! Saved {n_files} images as gif')

def folder_to_gif_stream(folder, gif_path, duration=0.5):
    file_iter = Path(folder).iterdir()
    n_files = len(os.listdir(folder))
    with iio.get_writer(gif_path, mode='I', duration=duration) as writer:
        for i, file in enumerate(file_iter):
            co(wait=True); print(f'{i+1}/{n_files}')
            if file.is_file():
                writer.append_data(iio.imread(file))
        writer.close()
#     optimize(gif_path)