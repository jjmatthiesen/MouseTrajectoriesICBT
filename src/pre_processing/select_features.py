from src.utils.imports import *

if __name__ == '__main__':
    # path where the features.csv is saved
    path_features = '../../data/features/'
    df = pd.read_csv(path_features + 'features_user_all.csv')
    df_selected = df[[
        'user_id',
        'participant_id',
        'assessment_id',
        'speed_avg',
        'angle_change_mean',
        'acute_angles',
        'obtuse_angles',
        'jitter',
        'pause_time_total',
        'moved_dist',
        'number_dp',
        'pauses_no',
        'scroll_speed_mean'
    ]]
    df_selected.to_csv(path_features + 'features_selected.csv')

