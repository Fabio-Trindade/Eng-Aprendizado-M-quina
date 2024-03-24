from src.utils.util_path import UtilPath


class KPaths:
    path_root = UtilPath.get_root_path()
    path_requirements = path_root + '/requirements'
    path_images = path_root + '/images'
    path_datasets = path_root + '/datasets'
    path_image_score_kaggle = path_images + '/score_kaggle_tp1.png'
    path_titanic_csv = path_datasets + '/titanic_train.csv'
    path_iris_json = path_datasets + '/iris.json'
    path_pib_mun_xls = path_datasets + '/PIB dos Municipios - base de dados 2010-2016.xls'
    path_champion_info_json = path_datasets + '/champion_info.json'
    path_user_track_parquet = path_datasets + '/user_tracking.parquet'
    path_churn_bank = path_datasets + '/ap3/Churn_Bank.csv'
    path_heart_csv = path_datasets + '/ap3/heart.csv'