from src.utils.util_path import UtilPath


class KPaths:
    path_root = UtilPath.get_root_path()
    path_requirements = path_root + '/requirements'
    path_images = path_root + '/images'
    path_image_score_kaggle = path_images + '/score_kaggle_tp1.png'