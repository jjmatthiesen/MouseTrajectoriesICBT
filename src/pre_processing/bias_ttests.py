import pandas as pd
from scipy import stats

PATH = '/path/to/data/directory/'  # Adapt this path to your project structure
df = 'dataset_version.csv'

# Split Baseline data into mouse and touch device users
touch = df[df.pre_mousedata==0]
mouse = df[df.pre_mousedata==1]


def tests_ind(var, touch, mouse):
    """

    :param var: Variable Name
    :param touch: DF filtered for touch device users
    :param mouse: DF filtered for mouse device users
    :return: All statistical results as indicated
    """
    touch = touch.dropna(subset=[var])
    mouse = mouse.dropna(subset=[var])
    # Levene test to check for equal variance
    stat, p = stats.levene(touch[var], mouse[var])
    # If Levene test indicated non-equal variance, Welch's test is used, otherwise Student's test is used
    if float(p) < 0.05:
        stat_t, p_t = stats.ttest_ind(touch[var], mouse[var], equal_var=False)
    else:
        stat_t, p_t = stats.ttest_ind(touch[var], mouse[var])
    return var, mouse[var].mean(), touch[var].mean(), stat, p, stat_t, p_t

results = pd.DataFrame(columns=['Mean Mouse Group','Mean Touch Group', 'Statistic Levene Test', 'p-value Levene Test','Statistic Student Test', 'p-value Student Test'])

for col in ['MADRS_sum_SCREEN', 'MADRS_sum_PRE','age', 'female']:
    var, mean_m, mean_t, stat, p, stat_t, p_t = tests_ind(col, touch, mouse)
    results.loc[var] = mean_m, mean_t, stat, p, stat_t, p_t