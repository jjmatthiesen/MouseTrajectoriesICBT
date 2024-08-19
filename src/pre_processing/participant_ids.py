import pandas as pd
import numpy as np
import src.utils.globals as ut_globals
from src.utils.imports import *


def get_df(PATH, test_user=None, overview_print=True, overview_export=False):
    """
    Gives overview of participant number per treatment, time point and mobile/desktop
    it requires matching CSVs for the mousevent and this SQL statement:

    mobile SELECT DISTINCT mousetracker_id FROM mousetracker_event WHERE eventname = 'touchstart';
    as mousetracker_event_mobile.csv

    Sometimes no data was tracked. select all ids from sessions with data:
    SELECT DISTINCT mousetracker_id FROM mousetracker_event WHERE eventname IS NOT NULL
    as mousetracker_event_active.csv

    Sometimes there is no mousemove recorded. And not even a touchstart for mobile users.
    Select all ids from session with mousemoves:
    SELECT DISTINCT mousetracker_id FROM mousetracker_event WHERE eventname = 'mousemove';
    as mousetracker_event_mousemove.csv

    :param PATH: Takes the path to the Kaldo folder on the KI server or local
    :param test_user: List of user IDs to delete from the df
    :param overview_print: If yes, the monthly participant counts per questionnaire are printed
    :return: df without the test users and with the additional columns 'assessment' and 'mobile'
    """
    if test_user is None:
        test_user = [3692506, 3692856, 3721999]
    df = pd.read_csv(PATH + 'mousetracker.csv')
    # Add the right names
    assessments = {1939484: 'DANA_Screen', 1939486: 'DANA', 1944974: 'DANA_XTRA', 2002265: 'SOPHIA_Screen',
                   1975153: 'MDD_A_SOPHIA', 1979094: 'MDD_B_SOPHIA', 1979156: 'PD_A_SOPHIA', 1979672: 'PD_B_SOPHIA',
                   1979941: 'SAD_A_SOPHIA', 1980219: 'SAD_B_SOPHIA'}

    df['assessment'] = df['assessment_id'].map(assessments)
    df.loc[:, 'time_point'] = np.where(
        np.logical_or((df['assessment'] == 'DANA_Screen'), (df['assessment'] == 'SOPHIA_Screen')), 'Screen', 'Pre')

    # all ids from sessions with data
    df_active = pd.read_csv(PATH + 'mousetracker_event_active.csv')
    df = df[df['id'].isin(df_active['mousetracker_id'])]

    # encompasses all users, which have a touch start
    df_mob = pd.read_csv(PATH + 'mousetracker_event_mobile.csv')
    df_mob['mobile'] = 'Mobile'
    df = df.merge(df_mob, how='left', left_on='id', right_on='mousetracker_id')
    df = df.drop(columns='mousetracker_id')

    # all users, which have at least one mousemove event
    df_mousemove = pd.read_csv(PATH + 'mousetracker_event_mousemove.csv')
    df_mousemove['device'] = 'Desktop'
    df = df.merge(df_mousemove, how='left', left_on='id', right_on='mousetracker_id')
    df = df.drop(columns='mousetracker_id')

    # touch devices can trigger a mouse event (e.g. mousedown on click)
    # They still do not deliver a mouse trajectory.
    # if the user had ones a touch event -> mobile
    df['mobile'] = df['mobile'].combine_first(df['device'])
    df = df.drop(columns='device')

    df['mobile'] = df['mobile'].fillna('not_used')
    sessions_not_used = df[df['mobile'] == 'not_used'][['id', 'participant_id']]
    # p-id from patients where some session where not used. It can be though that other sessions are useful.
    not_used_p_id = sessions_not_used.drop_duplicates(subset=['participant_id'], keep='first')

    # there are cases where a mobile-sized device had no touchstart and single mousemoves.
    # After manually inspection, it was clear that those are indeed mobile devices.
    # label all session with a device size of length 6 (e.g. 111x333) as mobile
    df.loc[df['screen_xy'].str.len() == 7, 'mobile'] = 'Mobile'

    df = df[~df['participant_id'].isin(test_user)]

    # Count individual patients per assessment
    df_participants_count = df.drop_duplicates(subset=['participant_id', 'assessment_id', 'time_point', 'mobile'], keep= 'first').copy(deep=True)
    df_participants_count_pre = df_participants_count[df_participants_count['time_point'] == "Pre"]
    df_participants_count_pre = df_participants_count_pre[df_participants_count_pre['mobile'] == "Desktop"]
    df_participants_count_pre = df_participants_count_pre.sort_values('participant_id', ascending=False)
    df_participants_count_screen = df_participants_count[df_participants_count['time_point'] == "Screen"]
    df_participants_count_screen = df_participants_count_screen[df_participants_count_screen['mobile'] == "Desktop"]
    # save lists
    pathlib.Path('../../data/preprocessed/investigation/').mkdir(parents=True, exist_ok=True)
    df_participants_count_pre['participant_id'].to_csv('../../data/preprocessed/list_of_desktop_participants_pre.csv', index=False)
    df_participants_count_screen['participant_id'].to_csv('../../data/preprocessed/list_of_desktop_participants_screen.csv', index=False)

    df_participants_count['created_at'] = pd.to_datetime(df_participants_count['created_at'])
    df_participants_count.loc[:, 'Month'] = df_participants_count['created_at'].dt.to_period('M')
    overview_part = pd.pivot_table(df_participants_count, values='participant_id', index='assessment_id',
                                   columns='Month', aggfunc='count')
    overview_part = overview_part.fillna(0).astype('int')
    overview_part.index = pd.Series(overview_part.index).map(ut_globals.assessments, na_action='ignore')

    if overview_print:
        print(overview_part)

    if overview_export:
        pathlib.Path('../../data/investigation/').mkdir(parents=True, exist_ok=True)
        overview_part.to_csv('../../data/investigation/participants_overview.csv')

    if df_participants_count['participant_id'].value_counts().max() != 2:
        print('There are patients with more than two questionnaires')

    print(pd.pivot_table(df_participants_count, values='participant_id', columns='mobile', index='time_point',
                         aggfunc='count'))

    m = df_participants_count['participant_id'].value_counts()
    two_sessions = m[m >= 2].index.to_list()
    df_two_sessions = df_participants_count[df_participants_count['participant_id'].isin(two_sessions)]

    df_two_sessions = df_two_sessions.sort_values('participant_id').reset_index()
    changed_to_mobile = 0
    changed_to_desktop = 0
    stayed_on_mobile = 0
    stayed_on_desktop = 0
    for ind in range(len(df_two_sessions) - 1):
        if ind % 2 == 0:
            if df_two_sessions['mobile'][ind] != df_two_sessions['mobile'][ind + 1]:
                if df_two_sessions['mobile'][ind] == 'Mobile':
                    changed_to_mobile += 1
                else:
                    changed_to_desktop += 1
            else:
                if df_two_sessions['mobile'][ind] == 'Mobile':
                    stayed_on_mobile += 1
                else:
                    stayed_on_desktop += 1
    print("changed to mobile:" + str(changed_to_mobile))
    print("changed to desktop:" + str(changed_to_desktop))
    print("stayed on mobile: " + str(stayed_on_mobile))
    print("stayed on desktop: " + str(stayed_on_desktop))

    pathlib.Path('../../data/investigation/').mkdir(parents=True, exist_ok=True)
    # df_participants_count.to_csv("../../data/investigation/df_participants_count.csv")
    return df


if __name__ == '__main__':
    path = "../../data/mouse_data/raw/"
    df_participants = get_df(path)
