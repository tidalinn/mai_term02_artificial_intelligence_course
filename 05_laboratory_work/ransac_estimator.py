'''RANSAC estimator module
'''

from data import PointGenerator
from ransac import RANSAC


def estimate_ransac(points__n_points: int, 
                    points__outliers_ratio: float,
                    points__k: float,
                    points__b: float,
                    points__epsilon: float,
                    ransac__n_iters: int,
                    ransac__in_ratio: float,
                    ransac__epsilon: float,
                    ransac__title: str, 
                    ransac__save: bool = True) -> None:
    
    '''Оценка принадлежности точек к линии

        Args:
            points__n_points (int): Количество точек
            points__outliers_ratio (float): Доля не принадлежащих к прямой точек
            points__k (float): Наклон прямой
            points__b (float): Сдвиг прямой
            points__epsilon (float): Порог, отсекающий принадлежность точек к прямой
            ransac__n_iters (int): Количество итераций
            ransac__in_ratio (float): Доля принадлежащих к прямой точек
            ransac__epsilon (float): Порог, отсекающий принадлежность точек к прямой
            ransac__title_suffix (str): Название графика
            ransac__save (str): Флаг сохранения графика
        '''
    
    # generate points
    generator = PointGenerator(
        n_points=points__n_points, 
        outliers_ratio=points__outliers_ratio
    )
    
    points = generator.generate_case(k=points__k, b=points__b, epsilon=points__epsilon)
    
    # estimate ransac
    ransac = RANSAC()
    
    ransac.clear_case()
    ransac.set_case(points, n_iters=ransac__n_iters, epsilon=ransac__epsilon)
    params = ransac.fit()
    
    ransac.draw(f'RANSAC estimation {ransac__title}', save=ransac__save)