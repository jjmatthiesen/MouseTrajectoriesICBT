from src.utils import utils as ut_utils
import src.utils.globals as ut_globals
import src.utils.plots as ut_plots
from src.utils.imports import *


def make_path_and_save_file(path, f_title):
    """
    :param path: the path for saving
    :param f_title: the file title
    :return:
    """
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    events.to_csv(path + f_title + '.csv', index=False)


if __name__ == '__main__':
    """
    pre-processing of the raw mouse data 
    - Sorts files into desktop and mobile user
    - within those folder, separate into train_test (user has label) and pre_train (user has no label)
    - within train_test into dropout or non_dropout according to the label
    - within dropout/non_dropout into Pre and Screening according to the time of recording
    - within these folders, a folder for each user with their sessions files are created
    
    |--Desktop
        |-- pre_train
            |--non_known
                |--pre
                    |--[user01ID]
                    |--[user02ID]
                    |--...
                |--screening
                    |--[user01ID]
                    |--[user02ID]
                    |--...
        |-- train_test
            |--dropout
                |--pre
                    |--[user01ID]
                    |--[user02ID]
                    |--...
                |--screening
                    |--[user01ID]
                    |--[user02ID]
                    |--...
            |--non_dropout
                |--pre
                    |--[user01ID]
                    |--[user02ID]
                    |--...
                |--screening
                    |--[user01ID]
                    |--[user02ID]
                    |--...
    |--Mobile
        | {same as above}
    """
    # ########################
    # Arguments
    save_as_img = False
    save_file = True
    folder_path = "../../data/"
    outcome_file = "20240703_outcomes.csv"
    # ########################

    df_p = pd.read_csv(folder_path + "/mouse_data/raw/mousetracker.csv")
    df_m = pd.read_csv(folder_path + "/mouse_data/raw/mousetracker_event.csv")
    no_data = 0
    list_of_desktop_participants_screen = []
    list_of_desktop_participants_pre = []
    list_to_investigate = []
    list_assessments = list(ut_globals.assessments.values())
    count_assessments = {k: 0 for k in list_assessments}

    # for every id in mousetracker.csv
    for i in range(len(df_p)):
        print(df_p.loc[i, 'id'])
        df_p_row = df_p.loc[i]
        df_p_id = df_p_row['id']
        df_p_pid = df_p_row['participant_id']
        has_touch_events = 'touchstart' in df_m.loc[df_m['mousetracker_id'] == df_p_id]['eventname'].values
        has_mousemove = 'mousemove' in df_m.loc[df_m['mousetracker_id'] == df_p_id]['eventname'].values
        # select all mouse data where the id (from participants df) == mousetracker_id
        events = df_m.loc[df_m['mousetracker_id'] == df_p_id]
        file_title = 'id' + str(df_p_id) + '__' + ut_globals.assessments[df_p_row['assessment_id']] + '_tID_' + str(
            df_p_row['assessment_id']) + "_pID_" + str(df_p_pid) + '_screen_' + str(
            df_p_row['screen_xy'])
        outcomes = pd.read_csv(folder_path + outcome_file)
        participants_dropout = np.array(outcomes[outcomes['dropout_mod'] == 1]['Internt ID'])
        participants_non_dropout = np.array(outcomes[outcomes['dropout_mod'] == 0]['Internt ID'])
        if df_p_pid in participants_dropout:
            s_path = "dropout/"
        elif df_p_pid in participants_non_dropout:
            s_path = "non_dropout/"
        else:
            s_path = "not_known/"
        # desktop users with actual coordinates some users just have the value 0
        if len(df_p_row['screen_xy']) > 7 and not has_touch_events and has_mousemove:
            if df_p_row['assessment_id'] == 1939484 or df_p_row['assessment_id'] == 2002265:
                if not (df_p_pid in list_of_desktop_participants_screen):
                    list_of_desktop_participants_screen.append(df_p_pid)
            else:
                if not (df_p_pid in list_of_desktop_participants_pre):
                    list_of_desktop_participants_pre.append(df_p_pid)
            if save_file:
                if df_p_row['assessment_id'] == 1939484 or df_p_row['assessment_id'] == 2002265:
                    if df_p_pid in np.array(outcomes['Internt ID']):
                        # train test split is split later
                        make_path_and_save_file(
                            folder_path + 'mouse_data/participants/desktop/train_test/' + s_path + 'Screening/' + str(df_p_pid) + '/',
                            file_title)
                    else:
                        make_path_and_save_file(
                            folder_path + 'mouse_data/participants/desktop/pre_train/' + s_path + 'Screening/' + str(df_p_pid) + '/',
                            file_title)
                else:
                    if df_p_pid in np.array(outcomes['Internt ID']):
                        make_path_and_save_file(
                            folder_path + 'mouse_data/participants/desktop/train_test/' + s_path + 'pre/' + str(df_p_pid) + '/',
                            file_title)
                    else:
                        make_path_and_save_file(
                            folder_path + 'mouse_data/participants/desktop/pre_train/' + s_path + 'pre/' + str(df_p_pid) + '/',
                            file_title)
            if save_as_img:
                events.loc[:, ['page_y']] = events.loc[:, ['page_y']] * -1
                movement = np.array(events[['eventtime', 'client_x', 'client_y', 'page_x', 'page_y', 'eventname', 'scrollspeed']])
                user_window, user_screen, user_document = ut_utils.get_view_size(df_p_row)
                s_path = folder_path + "mouse_data/participants/desktop_imgs/" + s_path
                pathlib.Path(s_path)
                # in one case the user document y value was smaller than the coordinates.
                # We make sure that the document size it at least the size of the coordinates
                if min(movement[:, 4]) < user_document[1]:
                    user_document[1] = min(movement[:, 4])
                ut_plots.plot_mouse(movement, user_document, coords='page', trajectory=True, clicks=True, title=file_title, save=True, show=False, save_path=s_path)
        # mobile users with touch data
        elif has_touch_events or len(df_p_row['screen_xy']) <= 7:
            if save_file:
                if df_p_row['assessment_id'] == 1939484 or df_p_row['assessment_id'] == 2002265:
                    make_path_and_save_file(
                        folder_path + 'mouse_data/participants/mobile/' + s_path + 'Screening/' + str(df_p_pid) + '/',
                        file_title)
                else:
                    make_path_and_save_file(
                        folder_path + 'mouse_data/participants/mobile/' + s_path + 'Pre/' + str(df_p_pid) + '/',
                        file_title)
        # users with just pause events or otherwise not useful
        else:
            if len(events) > 0 and save_file:
                if df_p_row['assessment_id'] == 1939484 or df_p_row['assessment_id'] == 2002265:
                    make_path_and_save_file(
                        folder_path + 'mouse_data/participants/not_used/' + s_path + 'Screening/' + str(df_p_pid) + '/',
                        file_title)
                else:
                    make_path_and_save_file(
                        folder_path + 'mouse_data/participants/not_used/' + s_path + 'Pre/' + str(df_p_pid) + '/',
                        file_title)
                print("not used " + str(df_p.loc[i, 'id']))
    print(list_of_desktop_participants_pre)
    folder_path = "../../data/mouse_data/participants/desktop/train_test/*/pre/*/"
    users = glob(folder_path)
    list_uf_users_files = []
    for u in users:
        list_uf_users_files.append(int(u.split("/")[-2]))
    for p in list_of_desktop_participants_pre:
        if p not in list_uf_users_files:
            print(p)


