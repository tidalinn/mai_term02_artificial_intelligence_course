'''Main module
'''

from ransac_estimator import estimate_ransac


def main():
    # test 1
    estimate_ransac(
        points__n_points=100, 
        points__outliers_ratio=0.1,
        points__k=1.5,
        points__b=2.0,
        points__epsilon=0.3,
        ransac__n_iters=1000,
        ransac__in_ratio=0.8,
        ransac__epsilon=0.35,
        ransac__title='01'
    )
    
    # test 2
    estimate_ransac(
        points__n_points=100, 
        points__outliers_ratio=0.5,
        points__k=1.5,
        points__b=1.,
        points__epsilon=0.8,
        ransac__n_iters=1000,
        ransac__in_ratio=0.5,
        ransac__epsilon=0.85,
        ransac__title='02'
    )
    
    # test 3
    estimate_ransac(
        points__n_points=80, 
        points__outliers_ratio=0.3,
        points__k=1.7,
        points__b=2.2,
        points__epsilon=0.5,
        ransac__n_iters=1000,
        ransac__in_ratio=0.7,
        ransac__epsilon=0.55,
        ransac__title='03'
    )
    
    
if __name__ == 'main':
    main()