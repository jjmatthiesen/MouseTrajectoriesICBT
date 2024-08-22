from src.utils.imports import *

# Update this import based on your project structure
from src.utils.utils import logger_nonseq
from src.utils.plots import plot_all_dropout
from src.utils.globals import assessments, arm_map


def id_in_id(df1: pd.DataFrame, column1: str, df2: pd.DataFrame, column2: str) -> tuple[list[str], list[str]]:
    """
    Compare unique IDs between two DataFrames and return IDs that are in both and those that are not.
    Used to see for which patients both outcome and mouse data is available.
    Args:
        df1 (pd.DataFrame): First DataFrame to compare.
        column1 (str): Column name in df1 containing the IDs.
        df2 (pd.DataFrame): Second DataFrame to compare.
        column2 (str): Column name in df2 containing the IDs.

    Returns:
        tuple: A tuple containing a list of IDs present in both DataFrames, and a list of IDs not present in both.
    """
    ids_kept = [id for id in df1[column1].unique() if id in df2[column2].unique()]
    ids_lost = ([id for id in df1[column1].unique() if id not in df2[column2].unique()]
                + [id for id in df2[column2].unique() if id not in df1[column1].unique()])
    return ids_kept, ids_lost


def get_mouse_times(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Merge mouse data times with the main DataFrame based on participant ID and assessment ID.

    Args:
        df1 (pd.DataFrame): Main DataFrame containing participant data.
        df2 (pd.DataFrame): DataFrame containing mouse tracking data.

    Returns:
        pd.DataFrame: Updated DataFrame with earliest and latest mouse data times merged.
    """
    # Convert 'created_at' to date and drop duplicates
    df2['created_at'] = pd.to_datetime(df2['created_at']).dt.date
    df2 = df2.drop_duplicates(subset=['assessment_id', 'participant_id', 'created_at']).copy(deep=True)
    df2.assessment_id = df2.assessment_id.map(assessments)

    # Identify Screening assessments and calculate earliest and latest mouse data times
    df2['Screening'] = np.where(df2.assessment_id.str.contains("Screening"), 1, 0)
    early_day = pd.pivot_table(df2, columns='Screening', index='participant_id', values='created_at', aggfunc='min')
    early_day.columns = ['earliest_mouse_pre', 'earliest_mouse_screen']
    df = df1.merge(early_day, left_on='Internt ID', right_on=early_day.index, how='left')

    late_day = pd.pivot_table(df2, columns='Screening', index='participant_id', values='created_at', aggfunc='max')
    late_day.columns = ['late_mouse_pre', 'late_mouse_screen']
    df = df.merge(late_day, left_on='Internt ID', right_on=late_day.index, how='left')

    return df

def del_test(df1: pd.DataFrame) -> pd.DataFrame:
    """
    VERY SPECIFIC TO THIS PROJECT:
    Remove test participants from the DataFrame based on 'Grupp' column values.

    Args:
        df1 (pd.DataFrame): DataFrame containing participant data.

    Returns:
        pd.DataFrame: Updated DataFrame without test participants.
    """
    df = df1[(df1['Grupp'] != 'Testgrupp') & (df1['Grupp'] != 'Test') & (df1['Grupp'] != 'TEST')].copy(deep=True)
    return df


def get_timeframe(df: pd.DataFrame, date_earl: str = '2023-03-01', date_late: str = '2024-03-31') -> pd.DataFrame:
    """
    Filter the DataFrame to include only participants whose assessment dates fall within the specified timeframe.

    Args:
        df (pd.DataFrame): DataFrame containing participant data.
        date_earl (str): Earliest date to include (default is '2023-03-01').
        date_late (str): Latest date to include (default is '2024-03-31').

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    # List of columns containing dates to filter on
    dates = ['Behandlingsstart', 'MADRS_SCREEN_1_DATE', 'MADRS_PRE_1_DATE', 'MADRS_POST_1_DATE', 'Behandlingsslut']

    # Convert columns to datetime format
    for col in dates:
        df[col] = pd.to_datetime(df[col], format='%Y-%m-%d')

    # Convert the input date strings to date objects
    date_earl = pd.to_datetime(date_earl, format='%Y-%m-%d').date()
    date_late = pd.to_datetime(date_late, format='%Y-%m-%d').date()

    # Filter the DataFrame based on the specified timeframe
    df = df[(df.MADRS_PRE_1_DATE.dt.date >= date_earl) | (df.earliest_mouse_pre >= date_earl)].copy(deep=True)
    df = df[(df.MADRS_PRE_1_DATE.dt.date <= date_late) | (df.earliest_mouse_pre <= date_late)].copy(deep=True)

    return df


if __name__ == '__main__':
    ### Parameters
    name_logger = '20240601_run'
    PATH = '/path/to/your/data/'  # Update this path accordingly
    outcome_file = '20240621_baseline_final.csv'
    cut_off_dropout_sophia = 7
    cut_off_dropout_dana = 6

    # Start logging
    logger = logger_nonseq(name_logger, level=2)

    # Load the main data file
    df1 = pd.read_csv(os.path.join(PATH, outcome_file))

    # Drop missing ID, test user, and non-starters
    df1 = df1.dropna(subset=['Internt ID'])
    df1 = del_test(df1)
    df1 = df1[~df1['Behandlingsstart'].isna()].copy(deep=True)

    # Get participant lists for pre and screen stages
    df_pre = pd.read_csv(os.path.join(PATH, 'list_of_desktop_participants_pre.csv'))
    df_screen = pd.read_csv(os.path.join(PATH, 'list_of_desktop_participants_screen.csv'))

    # Indicate participants with pre and screen mouse data
    df1.loc[:, 'pre_mousedata'] = np.where(df1['Internt ID'].isin(df_pre.participant_id.unique()), 1, 0)
    print(df1.pre_mousedata.value_counts())
    df1.loc[:, 'screen_mousedata'] = np.where(df1['Internt ID'].isin(df_pre.participant_id.unique()), 1, 0)
    del df_pre

    # Load and process mouse tracking data
    df2 = pd.read_csv(os.path.join(PATH, 'mousetracker.csv'))
    df1 = get_mouse_times(df1, df2)
    del df2

    # Filter the data based on the timeframe
    df1 = get_timeframe(df1)

    # Map treatment groups to normalized labels
    df1['Treatment_norm'] = df1.Behandling.map(arm_map)

    # Log the timeframe of mouse data collection
    logging.info(
        'Mouse data collected between {} and {}'.format(
            str(df1[df1['pre_mousedata'] == 1].earliest_mouse_pre.min()),
            str(df1[df1['pre_mousedata'] == 1].earliest_mouse_pre.max()))
    )

    # Define dropout based on module completion thresholds
    df1['dropout_s'] = np.where(df1['Antal moduler'] <= cut_off_dropout_sophia, 1, 0)
    df1['dropout_d'] = np.where(df1['Antal moduler'] <= cut_off_dropout_dana, 1, 0)
    df1['dropout_mod'] = np.where(df1['Treatment_norm'] == 'Dana', df1['dropout_d'], df1['dropout_s'])

    # Plot dropout rates for all participants
    plot_all_dropout(df1, cut_off_dropout_sophia, cut_off_dropout_dana)

    # Calculate dropout counts by treatment group
    dropout_counts = pd.pivot_table(df1, columns='dropout_mod', index='Treatment_norm', values='Internt ID',
                                    aggfunc='count')

    # Calculate dropout counts for participants with pre mouse data
    dropout_counts_pre = pd.pivot_table(df1[df1['pre_mousedata'] == 1], columns='dropout_mod', index='Treatment_norm',
                                        values='Internt ID', aggfunc='count')

    # Log various statistics about the data
    logger.info(df1.Treatment_norm.value_counts())
    logger.info(dropout_counts[1] / dropout_counts.sum(axis=1))
    logger.info(df1['dropout_mod'].value_counts(normalize=True))
    logger.info(df1[df1['pre_mousedata'] == 1].Treatment_norm.value_counts())
    logger.info(dropout_counts_pre[1] / dropout_counts_pre.sum(axis=1))
    logger.info(df1[df1['pre_mousedata'] == 1]['dropout_mod'].value_counts(normalize=True))

    # Indicate gender and correct age if necessary
    df1['female'] = np.where(df1['Gender'] == 'Kvinna', 1, 0)
    df1['age'] = np.where(df1['age'] < 0, 48, df1['age'])  # Correct invalid ages
    df1['age'] = df1['Age'].astype('int')

    # Select columns to keep for the final output
    columns_keep = ['Internt ID', 'dropout_mod', 'MADRS_sum_SCREEN', 'MADRS_sum_PRE',
                    'Behandling', 'Treatment_norm', 'age', 'female', 'MADRS_PRE_1_DATE',
                    'pre_mousedata', 'screen_mousedata']

    # Save the final processed data to a CSV file
    df1[columns_keep].to_csv(os.path.join(PATH, 'YYYYMMDD_outcomes.csv'), index=False)