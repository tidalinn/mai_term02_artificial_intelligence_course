'''Data generation module
'''

import numpy as np


class PointGenerator:

    def __init__(self, n_points: int, outliers_ratio: float) -> None:
        '''
        Args:
            n_points (int): Количество точек
            outliers_ratio (float): Доля не принадлежащих к прямой точек
        '''
        
        self.n_points = n_points
        self.outliers_ratio = outliers_ratio
        
        self.n_inliers = int(np.ceil(n_points * (1 - outliers_ratio)))
        self.n_outliers = int(np.ceil(n_points * outliers_ratio))
        
        
    def generate_case(self, k: float = 1., b: float = 0., epsilon: float = 0.1) -> np.ndarray:
        '''Генерация точек выборки вместе с шумовыми для прямой y = kx + b + epsilon

        Args:
            k (float, optional): Наклон прямой. Defaults to 1
            b (float, optional): Сдвиг прямой. Defaults to 0
            epsilon (float, optional): Порог, отсекающий принадлежность точек к прямой. Defaults to 0.1
        
        Returns:
            np.ndarray: Обучающая выборка
        '''
        
        if k is None:
            k = np.random.uniform(-1, 1)

        if b is None:
            b = np.random.uniform(0, 5)   

        x = np.linspace(0, 10, self.n_inliers + 1)
        y = k * x + b + np.random.normal(scale=epsilon, size=len(x))
        
        inliers = np.vstack((x, y)).T # inliers

        x = np.random.uniform(0, 10, self.n_outliers)
        y = np.random.uniform(y.min(), y.max(), self.n_outliers)

        outliers = np.vstack((x, y)).T # outliers

        data = np.concatenate((inliers, outliers))
        np.random.shuffle(data)

        return data