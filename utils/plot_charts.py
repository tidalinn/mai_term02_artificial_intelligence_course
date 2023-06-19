'''charts plotting module
'''

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import contextlib
from PIL import Image

from tqdm.notebook import tqdm
from typing import List, Dict
import random
import glob
import os
import pathlib

from mlxtend.plotting import plot_decision_regions


# HEX colors generator
def generate_hex(n_colors: int, k: int = 6) -> List[str]:
    colors = []
    generate_hex_color = lambda hex_len: ''.join(random.choices('1234567890ABCDEF', k=hex_len))
    
    for _ in range(n_colors):
        hex_color = generate_hex_color(k)
        
        while hex_color in colors:
            print('oops')
            hex_color = generate_hex_color(k)
        
        colors.append('#' + hex_color)
    
    return colors


# multi-class plotting
def plot_class(data: pd.DataFrame, title: str) -> None:
    font_s = 12
    plt.figure(figsize=(6, 5))
    
    plt.title(f'{title}\n', fontsize=font_s)
    
    n_classes = len(set(data['class']))
    hex_colors = generate_hex(n_classes)
    
    for i in range(n_classes):
        data_i = data[data['class'] == i]
        
        plt.scatter(data_i['feature_1'], 
                    data_i['feature_2'],
                    c=hex_colors[i],
                    edgecolors='black',
                    linewidths=0.5,
                    label=f'class {i}')
        
    plt.xlabel('feature 1', fontsize=font_s)
    plt.ylabel('feature 2', fontsize=font_s)
    
    plt.legend(loc='upper right')
    
    plt.grid()
    plt.show()
    
    
# correlation matrix plotting
def plot_corr(data: pd.DataFrame) -> None:
    font_s = 12
    plt.figure(figsize=(6, 5))

    corr_map = sns.heatmap(data.corr(), 
                           annot=True, 
                           fmt='.1g', 
                           cmap='coolwarm', 
                           annot_kws={'fontsize': font_s-3})

    plt.show()
    
    
# directories existance checking
def check_dir(paths: List[str] = ['visualization/', 'images/', 'gifs/']) -> List[str]:
    path = paths[0]
    
    if os.path.isdir(path) == False:
        os.mkdir(path)
        print(f'Created directory {path}')
    
    for i in range(1, len(paths)):
        sub_path = f'{path}{paths[i]}'
        
        if os.path.isdir(sub_path) == False:
            os.mkdir(sub_path)
            print(f'Created directory {sub_path}')
    
    return paths
            
            
# GIF creation
def create_gif(title: str, img_path: str, gif_path: str) -> None:
    
    images_path = f'{img_path}{title}_*.png'
    gif_path = f'{gif_path}{title}.gif'

    with contextlib.ExitStack() as stack:
        images_stack = (stack.enter_context(Image.open(image))
                        for image in sorted(glob.glob(images_path)))

        image = next(images_stack)

        image.save(fp=gif_path, 
                   format='GIF', 
                   append_images=images_stack,
                   save_all=True, 
                   duration=250, 
                   disposal=2,
                   loop=0)
        
        
# GIF printing
def print_gif(gif_name: str, gif_path: str) -> None:
    from IPython.display import Image

    with open(f'{gif_path}{gif_name}.gif', 'rb') as file:
        display(Image(file.read()))
        
        
# decision regions GIF creation
def create_decision_regions_gif(title: str,
                                models: list = None,
                                X: np.ndarray = None,
                                Y: np.ndarray = None,
                                params: List[str] or Dict[str, str or float or int] = None,
                                value: float = 1.5,
                                width: float = 3.,
                                images_saved: bool = False) -> None:
    
    # creating directories
    visual_dir, img_dir, gif_dir = check_dir()
    
    file_title = title.lower().replace(' ', '_')
    
    if images_saved == False:
        for i in tqdm(range(len(models)), 'Creating GIF'):
            save_plot(i, models[i], X, Y, params, title)
    
    
    # create gif
    create_gif(file_title, 
               os.path.join(visual_dir, img_dir), 
               os.path.join(visual_dir, gif_dir))
    
    
    # print gif
    print_gif(file_title, os.path.join(visual_dir, gif_dir))
    

# plot saving
def save_plot(index: int,
              model, 
              X: np.ndarray,
              Y: np.ndarray,
              params: List[str] or Dict[str, str or float or int],
              title: str,
              value: float = 1.5,
              width: float = 3.,
              show: bool = False) -> None:
    
    # creating directories
    visual_dir, img_dir, gif_dir = check_dir()
    
    file_title = title.lower().replace(' ', '_')
    
    font_s = 12
    fig, ax = plt.subplots()
    
    try:
        title_add = '\n'.join(['{0} {1:03}'.format(param, model.get_params()[param]) 
                               if type(model.get_params()[param]) in [int, float] else
                               '{0} {1}'.format(param, model.get_params()[param])
                               for param in params])
    except:
        key, val = list(params.items())[0]
        title_add = f'{key} {val:.03}'
        
    fig.suptitle(title + ' ' + title_add, fontsize=font_s+4)

    plot_decision_regions(X, 
                          Y, 
                          clf=model,
                          filler_feature_values={2: value, 3: value, 4: value},
                          filler_feature_ranges={2: width, 3: width, 4: width},
                          legend=2, 
                          ax=ax)

    file_name = title.lower().replace(' ', '_') + '_{:03}.png'.format(index)
    file_path = os.path.join(visual_dir, img_dir, file_name)

    plt.legend(loc='upper right')

    plt.savefig(file_path)
    
    if show:
        plt.show()
    else:
        plt.close()
        
        
# single value chart plotting
def plot_value(value: np.ndarray, title: str, x_axis: str, y_axis: str) -> None:
    font_s = 12
    plt.figure(figsize=(6, 5))
    
    plt.title(f'{title}\n', fontsize=font_s)
    
    plt.plot(value)
    
    plt.xlabel(x_axis.lower(), fontsize=font_s)
    plt.ylabel(y_axis.lower(), fontsize=font_s)

    plt.grid()
    plt.show()
    
    
# random image from a dataset plotting
def plot_random_image(target_dir: str,
                      seed: int = None,
                      depth: str = '*/*/*') -> None:
    
    random.seed(seed)

    image_paths = list(pathlib.Path(target_dir).glob(f'{depth}.jpg'))
    image_random = random.choice(image_paths)
    image_class = image_random.parent.stem
    
    img = Image.open(image_random)
    
    print('Путь к изображению:', image_random)
    print('Класс изображения:', image_class)
    print(f'Высота: {img.height} | Ширина: {img.width}')
    
    img = np.array(img)
    plt.imshow(img)
    
    print(f'Размерность изображения {img.shape} -> [height, width, color_channels]')