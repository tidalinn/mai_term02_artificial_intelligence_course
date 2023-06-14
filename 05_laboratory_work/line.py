'''Lines generation module
'''

import numpy as np
from typing import Tuple


class Line():

    def __init__(self, points: np.ndarray) -> None:
        self.points = points
        
        self.k = None
        self.b = None
        

    def estimate_params(self, shift: float = 0.000001) -> None:
        '''Оценка параметров прямой по двум точкам

        Args:
            shift (float, optional): Cдвиг прямой. Defaults to 0.000001
        '''
        
        n_points = len(self.points)

        if n_points > 2:
            raise NotImplementedError
        
        elif n_points < 2:
            raise ValueError(f'Not enough points. Must be at least 2, but got {n_points}')
        
        else:
            x1, y1 = self.points[0]
            x2, y2 = self.points[1]

            self.k = (y1 - y2) / (x1 - x2 + shift)
            self.b = y2 - self.k * x2 # line by 2 dots
            

    def divide_points(self, 
                      points: np.ndarray, 
                      in_out_threshold: float = 0.1, 
                      shift: float = 0.000001) -> Tuple[np.ndarray, np.ndarray]: 
        
        '''Разделение точек по принадлежности к прямой

        Args:
            points (np.ndarray): Массив координат точек
            in_out_threshold (float, optional): Порог, отсекающий принадлежность точек к прямой. Defaults to 0.1
            shift (float, optional): Дополнительный коэффициент. Defaults to 0.000001
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: принадлежащие к прямой точки и не принадлежащие
        '''
        
        self.estimate_params(shift) # estimate params

        distance = np.abs(self.k * points[:, 0] - points[:, 1] + self.b) / np.sqrt(self.k ** 2 + 1 + shift)
        
        inliers = points[distance <= in_out_threshold]
        outliers = points[distance > in_out_threshold]
        
        return inliers, outliers