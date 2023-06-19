'''useful functions module
'''

from typing import List
import os
import random


# folders content counting
def walk_through_dir(target_dir: str) -> None:
    counter, total_dir, total_img = 1, 0, 0
    
    for dirpath, dirnames, filenames in os.walk(target_dir):
        dirpath = dirpath.replace('\\', '/')
        label = dirpath.split('/')[-1]
        
        if label in target_dir:
            space = ''
            
            print(f'{space}Путь: {dirpath} -> Каталогов: {len(dirnames)}\n')
        
        elif label in ['train', 'test']:
            space, end = '', ''
            total_dir += len(dirnames)
            
            print(f'\t{space}Каталог {label} -> Каталогов: {len(dirnames)}{end}')
            
        else:
            space, end = '├── ', ''
            
            if counter == total_dir:
                space, end = '└── ', '\n'
            
            print(f'\t{space}Класс {label} -> Изображений: {len(filenames)}{end}')
            counter += 1
        
        total_img += len(filenames)
        
        
# random image src generation
def get_random_images(path: str, n_images: int = 1) -> List[str]:
    images = []
    
    for i in range(n_images):
        random_label = random.choice(os.listdir(path))
        random_path = os.path.join(path, random_label)

        random_image = random.choice(os.listdir(random_path))
        image_path = os.path.join(random_path, random_image)
        
        images.append(image_path)
    
    return images