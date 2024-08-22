from src.utils.imports import *

def missing_madras(x_train: np.ndarray, x_test: np.ndarray, column_names: list[str]) -> tuple[
    pd.DataFrame, pd.DataFrame]:
    """
    Handle missing MADRS (Montgomery-Åsberg Depression Rating Scale) values in the training and test datasets.

    Args:
        x_train (np.ndarray): Training data array.
        x_test (np.ndarray): Test data array.
        column_names (list[str]): List of column names corresponding to the datasets.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: DataFrames for the training and test data with missing MADRS values handled.
    """
    # Convert arrays to DataFrames with appropriate column names
    x_train = pd.DataFrame(x_train, columns=column_names)
    x_test = pd.DataFrame(x_test, columns=column_names)

    # Check for missing values in the training data and log them
    if x_train.isna().sum().sum() > 0:
        logging.info('Missing values in training data: {}'.format(
            str([(col, x_train[col].isna().sum()) for col in x_train.columns if x_train[col].isna().sum() > 0])))

        # Create a flag for missing MADRS values
        x_train['MADRAS_missing'] = np.where(
            (x_train['MADRS_sum_PRE'].isna()) | (x_train['MADRS_sum_SCREEN'].isna()), 1, 0
        )
        x_test['MADRAS_missing'] = np.where(
            (x_test['MADRS_sum_PRE'].isna()) | (x_test['MADRS_sum_SCREEN'].isna()), 1, 0
        )

        # Calculate factors for imputing missing MADRS values
        factor_madras_screen = ((x_train.MADRS_sum_SCREEN - x_train.MADRS_sum_PRE) / x_train.MADRS_sum_SCREEN).mean()
        factor_madras_pre = ((x_train.MADRS_sum_SCREEN - x_train.MADRS_sum_PRE) / x_train.MADRS_sum_PRE).mean()

        # Impute missing MADRS values in the training data
        x_train = missing_madras_calc(x_train, factor_madras_screen, factor_madras_pre)

    # Check for missing values in the test data and log them
    if x_test.isna().sum().sum() > 0:
        logging.info('Missing values in test data: {}'.format(
            str([(col, x_test[col].isna().sum()) for col in x_test.columns if x_test[col].isna().sum() > 0])))

        # Impute missing MADRS values in the test data using factors from the training data
        x_test = missing_madras_calc(x_test, factor_madras_screen, factor_madras_pre)

    return x_train, x_test


def missing_madras_calc(df1: pd.DataFrame, factor_madras_screen: float, factor_madras_pre: float) -> pd.DataFrame:
    """
    Impute missing MADRS values in the DataFrame based on calculated factors.

    Args:
        df1 (pd.DataFrame): DataFrame containing participant data.
        factor_madras_screen (float): Factor for imputing missing MADRS_screen values.
        factor_madras_pre (float): Factor for imputing missing MADRS_pre values.

    Returns:
        pd.DataFrame: Updated DataFrame with imputed MADRS values.
    """
    # Impute missing MADRS_screen values based on MADRS_pre
    df1.MADRS_sum_SCREEN = round(df1.MADRS_sum_SCREEN.fillna(df1.MADRS_sum_PRE * factor_madras_screen))

    # Impute missing MADRS_pre values based on MADRS_screen
    df1.MADRS_sum_PRE = round(df1.MADRS_sum_PRE.fillna(df1.MADRS_sum_SCREEN * factor_madras_pre))

    # If there are still missing values, fill them with 0 and log the action
    if df1.isna().sum().sum() > 0:
        logging.info('All remaining missing features, except MADRS, were filled with 0')
        df1 = df1.fillna(0)

    return df1


def prep(PATH: str, file_name: str, y_name: str, x_name: list[str], mouse: bool = False, mouse_only: bool = False) -> \
tuple[pd.DataFrame, pd.Series]:
    """
    Prepare the dataset for analysis by loading data, handling mouse data, and selecting features.

    Args:
        PATH (str): Path to the directory containing the data files.
        file_name (str): Name of the CSV file to load.
        y_name (str): Name of the target variable column.
        x_name (list[str]): List of feature column names to include in the analysis.
        mouse (bool or str): If True, load and merge mouse data features. If a string, specify the mouse feature file.
        mouse_only (bool or list[str]): If True, use only mouse data features. If a list, specify specific columns.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature DataFrame and target variable Series.
    """
    # Load the main data file
    df = pd.read_csv(PATH + file_name)

    if mouse:
        # Filter for participants with pre-treatment mouse data
        df = df[df['pre_mousedata'] == 1].copy(deep=True)
        df_mouse = pd.read_csv(PATH + 'features/' + mouse)

        # Remove any erroneous duplicate headers and convert IDs to integers
        df_mouse = df_mouse[df_mouse['participant_id'] != 'participant_id'].copy(deep=True)
        df_mouse.participant_id = df_mouse.participant_id.astype('int')
        df_mouse.assessment_id = df_mouse.assessment_id.astype('int')
        df['Internt ID'] = df['Internt ID'].astype('int')

        # Exclude specific assessment IDs related to pre-treatment
        df_mouse = df_mouse[~df_mouse['assessment_id'].isin([1939484, 2002265])]
        missing = [id for id in df['Internt ID'] if id not in df_mouse['participant_id']]

        # Merge the main data with the mouse data
        df = df.merge(df_mouse, right_on='participant_id', left_on='Internt ID', how='left')

    # Select features based on mouse_only flag
    if mouse_only:
        x = df[mouse_only].copy(deep=True)
    else:
        x = df[x_name].copy(deep=True)

        # Create dummy variables for the treatment group if needed
        if 'Treatment_norm' in x_name:
            x = pd.concat([x, pd.get_dummies(x['Treatment_norm'])], axis=1)
            x = x.drop(columns=['Treatment_norm'])

    # Extract the target variable
    y = df[y_name].copy(deep=True)

    return x, y