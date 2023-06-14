'''RANSAC for 2D lines:

Algo:
    I STAGE (Hypothesis generation stage):
        1. Sample 2D points (1 simple: 2 points; 2 hard: 5 points)
        2. Model estimation (2 points: analytics; 5 points: MSE estimation) # mse estimator for a line from sklearn
    
    II STAGE (Hypothesis evaluation stage):
        3. Inlier counting (criterion: % inlier >  threshold [ex: 80% inliers of data]) 
           if True -> best params
           if False -> Step 1.
        4. # iterations > num_iter ? : Step 1.
        
'''

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import os

from line import Line


class RANSAC:

    def __init__(self) -> None:
        self.n_iters: int = 0
        self.in_ratio: float = 0.0
        self.epsilon: float = 0.0
        
        self.best_params: dict = {} # kx + b -> k, b
        self.inliers: list = [] # final to plot line
        self.outliers: list = [] # highlight
        self.points: np.ndarray = np.ndarray([])
        self.best_params: Tuple[float, float] = () # best found
            
    
    def set_case(self, 
                 points: np.ndarray, 
                 n_iters: int = 100, 
                 in_ratio: float = 0.8, 
                 epsilon: float = 0.1) -> None:
        
        '''Устанавливает массив точек и проверяет, нужная ли у него размерность

        Args:
            points (np.ndarray): Массив координат точек
            n_iter (float, optional): Количество итераций. Defaults to 100
            in_ratio (float, optional): Доля принадлежащих к прямой точек. Defaults to 0.8
            epsilon (float, optional): Порог, отсекающий принадлежность точек к прямой. Defaults to 0.1
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: принадлежащие к прямой точки и не принадлежащие
        '''

        self.points = points
        self.n_iters = n_iters
        self.in_ratio = in_ratio
        self.epsilon = epsilon
        

    def clear_case(self) -> None:
        '''Чистит все параметры, если нужно переобучить
        '''
        self.best_params = {}
        self.inliers = []
        self.outliers = []
        self.points = np.ndarray([])
        self.best_params = ()
        

    def fit(self) -> Tuple[float, float]:
        '''Оценка принадлежности точек к прямой по переданным параметрам
        
        Returns:
            Tuple[float, float]: параметры прямой (наклон и сдвиг)
        '''
        
        for i in range(self.n_iters):
            random_points_indexes = np.random.randint(0, len(self.points), 2) # get random dots
            
            point_1 = self.points[random_points_indexes[0]]
            point_2 = self.points[random_points_indexes[1]]
            
            line = Line(np.array([point_1, point_2])) # create line
            
            inliers, outliers = line.divide_points(self.points, self.epsilon) # divide points
            
            out_ratio = len(outliers) / len(self.points)
            
            if not (len(self.inliers) == 0 and len(self.outliers) == 0):
                out_ratio_update = len(outliers) / len(self.points)
                
                if out_ratio_update < out_ratio:
                    continue
                    
            self.best_params = [line.k, line.b]

            self.inliers = inliers
            self.outliers = outliers
            
            if (1 - out_ratio) > self.in_ratio:
                break

        return self.best_params
    

    def draw(self, title: str, save: bool = True) -> None:
        '''Отрисовка разброса точек и полученной прямой

        Args:
            title (str): Название графика
            save (bool, optional): Флаг сохранения графика. Defaults to False
        '''
        
        font_s = 12
        plt.figure(figsize=(6, 5))
        
        plt.title(f'{title}\n', fontsize=font_s+4)
        
        plt.scatter(self.inliers[:, 0], self.inliers[:, 1], c='blue', label='Inliers')
        plt.scatter(self.outliers[:, 0], self.outliers[:, 1], c='red', label='Outliers')

        x_min = min(self.inliers[:, 0].min(), self.outliers[:, 0].min())
        x_max = max(self.inliers[:, 0].max(), self.outliers[:, 0].max())
        
        x_line = np.linspace(x_min, x_max, 2)
        
        k, bias = self.best_params
        y_line = k * x_line + bias
        
        plt.plot(x_line, y_line, c='green', label='Estimated line')
        
        plt.xlabel('X', fontsize=font_s)
        plt.ylabel('Y', fontsize=font_s)
        
        plt.legend(loc='upper right')
        
        plt.grid()
        
        if save:
            path = 'images/'
            title = '_'.join(title.lower().split())

            if os.path.isdir(path) == False:
                os.mkdir(path)

            plt.savefig(f'{path}{title}.png', dpi=300)
        
        plt.show()