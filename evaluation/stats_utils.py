import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas


def t_test_x_differnt_y(x, y, x_label, y_label, output_path=None, df_file=None): # two sided
    df = pandas.DataFrame()
    df = df.append(is_normal(x, x_label, output_path))
    df = df.append(is_normal(y, y_label, output_path))

    stat, p = stats.ttest_ind(x, y, equal_var=False, alternative='two-sided')
    h0 = x_label + ' is not different ' + y_label
    df = df.append(pandas.DataFrame({'h0': [h0], 'test': ['t_test_two_sided'], 'stat': [stat], 'p': [p]}))
    if output_path and df_file:
        df.to_csv(output_path + df_file)
    return df


def t_test_x_greater_y(x, y, x_label, y_label, output_path=None, df_file=None): # one sided, x greater y
    df = pandas.DataFrame()
    df = df.append(is_normal(x, x_label, output_path))
    df = df.append(is_normal(y, y_label, output_path))

    stat, p = stats.ttest_ind(x, y, equal_var=False, alternative='greater')
    h0 = x_label + ' is not greater than ' + y_label
    df = df.append(pandas.DataFrame({'h0': [h0], 'test': ['t_test_one_sided'], 'stat': [stat], 'p': [p]}))
    if output_path and df_file:
        df.to_csv(output_path + df_file)
    return df


def is_normal(series, add_label, output_path=None):
    if output_path:
        stats.probplot(series, dist="norm", plot=plt)
        plt.savefig(output_path + add_label + '_normality.png')
        plt.close()
    shapiro_stat, shapiro_p = stats.shapiro(series)
    dagostino_stat, dagostino_p = stats.normaltest(series)

    df = pandas.DataFrame({'test': ['shapiro_stat', 'dagostino_stat'], 'stat': [shapiro_stat, dagostino_stat], 'p': [shapiro_p, dagostino_p]})
    df['h0'] = add_label + ' - that the data was drawn from normal distribution'
    return df

#
# def bootstrap_survey_submissions(survey_df, rater_1_df):
#     r1_df = rater_1_df.copy()
#     s_df = survey_df.copy()
#     r1_df['RootCause'] = r1_df['RootCause'].replace(target_names)
#     s_df['RootCause'] = s_df['RootCause'].replace(target_names)
#     s_df.describe().to_csv(OUTPUT_DIR + 'survey_describe.csv')
#
#     submission_scores_df = pandas.DataFrame()
#     for submission_id in s_df['submission_id'].value_counts().index.to_list():
#         submission_df = s_df[s_df['submission_id'] == submission_id].copy()
#         r1_target, r2_target = map_raters(r1_df, submission_df, remove_nan=True)
#
#         report = pandas.DataFrame(report_classifier_performance(r1_target, r2_target, list(target_names.keys()), 0, {}, submission_id))
#         submission_scores_df = submission_scores_df.append(pandas.DataFrame(report))
#
#     boxes = pandas.DataFrame()
#     boxes = boxes.append(do_boostrap(submission_scores_df['F1 weighted average'], ' weighted\naverage'))
#     boxes = boxes.append(do_boostrap(submission_scores_df['F1_concurrency'], 'con.'))
#     boxes = boxes.append(do_boostrap(submission_scores_df['F1_memory'], 'mem.'))
#     boxes = boxes.append(do_boostrap(submission_scores_df['F1_other'], 'oth.'))
#     boxes = boxes.append(do_boostrap(submission_scores_df['F1_semantic'], 'sem.'))
#
#     fig, ax = plt.subplots(figsize=(5, 4))
#     plot_bootstrap_boxdiagram(fig, ax, "", "F1", boxes) #, widths=(0.6, 0.6, 0.6, 0.6)
#     # plt.xticks(rotation=45)
#     plt.tight_layout()
#     # plt.ylim([0.4, 0.87])
#     plt.savefig(OUTPUT_DIR + 'survey_class_specific_f1_bootsrap_boxplot_dense.pdf')
#     boxes.to_csv(OUTPUT_DIR + 'survey_class_specific_f1_bootsrap_boxplot_dense.csv')
#     plt.close()


# def do_boostrap(series, label):
#     repl = [np.mean(np.random.choice(series.dropna(), size=len(series))) for x in range(0, 1000)]
#     return evaluate_bootstrap(np.array(repl), label)

def evaluate_bootstrap(series, label):
    mean = series.mean()
    alpha = 0.95
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(series, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(series, p))

    return {'alpha': alpha * 100,
                           'lower': lower * 100,
                           'upper': upper * 100,
                           'mean': mean,
                           'label': label}


def get_box(bootstrap_dict):
    return {
        'label': bootstrap_dict['label'],
        'whislo': bootstrap_dict['lower'] / 100,
        'q1': bootstrap_dict['lower'] / 100,
        'med': bootstrap_dict['mean'],
        'q3': bootstrap_dict['upper'] / 100,
        'whishi': bootstrap_dict['upper'] / 100,
        'fliers': []
    }
