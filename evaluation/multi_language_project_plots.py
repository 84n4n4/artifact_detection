import random
import traceback

import numpy as np
import pandas
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import LinearSVC

from artifact_detection_model.model_training import run_ml_artifact_training
from artifact_detection_model.utils.Logger import Logger
from datasets.constants import LANGUAGES
from datasets.dataset_utils import get_trainingset, get_all_validation_sets, get_validation_sets_for_language
from evaluation.stats_utils import evaluate_bootstrap, t_test_x_greater_y, t_test_x_differnt_y
from evaluation.utils import validation_performance_on_dataset
from file_anchor import root_dir

import seaborn as sns

log = Logger()

OUT_PATH = root_dir() + 'evaluation/out/multi_language/'
CROSS_LANGUAGE_EVALUATION = root_dir() + 'evaluation/out/cross_language/'

language_labels = {
    'cpp': 'C++',
    'java': 'Java',
    'javascript': 'JavaScript',
    'php': 'PHP',
    'python': 'Python',
}


def main():
    # cross_project_roc_auc_matrix('1')
    bare_stats()
    for validation_set_no in ['1', '2']:
        # cross_project_roc_auc_matrix(validation_set_no)
        # p_test_single_lang_model_performs_better_than_multi_lang_model(validation_set_no)
        # p_test_single_lang_model_different_than_multi_lang_model(validation_set_no)
        # p_test_single_lang_model_different_than_multi_lang_model(validation_set_no)
        roc_auc_boxplots(validation_set_no)

def bare_stats():
    multi_df = pandas.read_csv(OUT_PATH + 'artifact_detection_multi_language_model_resample_summary.csv')
    for lang in LANGUAGES:
        columns = ['roc-auc_' + lang + '_researcher_' + x for x in ['1', '2']]
        multi_df[columns].describe().to_csv(OUT_PATH + lang + '_performance.csv')


def p_test_single_lang_model_performs_better_than_multi_lang_model(validation_set_no):
    multi_df = pandas.read_csv(OUT_PATH + 'artifact_detection_multi_language_model_resample_summary.csv')

    rep_df = pandas.DataFrame()
    for lang in LANGUAGES:
        df = pandas.read_csv(CROSS_LANGUAGE_EVALUATION + lang + '_artifact_detection_cross_language_resample_summary.csv')

        rep = t_test_x_greater_y(df['roc-auc_' + lang + '_researcher_' + validation_set_no],
                                 multi_df['roc-auc_' + lang + '_researcher_' + validation_set_no],
                                 lang, 'multilang')  # one sided, x greater y
        rep['model'] = 'multilang'
        rep_df = rep_df.append(rep)

    rep_df.to_csv(OUT_PATH + 'single_lang_model_better_on_its_own_language_than_multilang_model_VS' + validation_set_no + '.csv')


def p_test_single_lang_model_different_than_multi_lang_model(validation_set_no):
    multi_df = pandas.read_csv(OUT_PATH + 'artifact_detection_multi_language_model_resample_summary.csv')

    rep_df = pandas.DataFrame()
    for lang in LANGUAGES:
        df = pandas.read_csv(CROSS_LANGUAGE_EVALUATION + lang + '_artifact_detection_cross_language_resample_summary.csv')

        rep = t_test_x_differnt_y(df['roc-auc_' + lang + '_researcher_' + validation_set_no],
                                 multi_df['roc-auc_' + lang + '_researcher_' + validation_set_no],
                                 lang, 'multilang')  # one sided, x greater y
        rep['model'] = 'multilang'
        rep_df = rep_df.append(rep)

    rep_df.to_csv(OUT_PATH + 'single_lang_model_than_multilang_model_VS' + validation_set_no + '.csv')


def cross_project_roc_auc_matrix(validation_set_no):
    columns = ['roc-auc_' + x + '_researcher_' + validation_set_no for x in LANGUAGES]
    cm = []
    for lang in LANGUAGES:
        df = pandas.read_csv(CROSS_LANGUAGE_EVALUATION + lang + '_artifact_detection_cross_language_resample_summary.csv')
        df = df[columns].mean()
        cm.append(df.to_list())

    df = pandas.read_csv(OUT_PATH + 'artifact_detection_multi_language_model_resample_summary.csv')
    df = df[columns].mean()
    cm.append(df.to_list())

    # disp = plot_numpy_confusion_matrix(cm, [language_labels[x] for x in LANGUAGES])
    # disp.ax_.set(ylabel="Model language", xlabel="Validation set 1 language", title='ROC-AUC')
    # plt.savefig(OUT_PATH + 'cross_project_roc_auc_matrix_VS' + validation_set_no + '.png')

    fig, ax = plt.subplots() #figsize=(3, 3)
    sns.heatmap(cm,
                ax=ax,
                # linewidths=0.01,
                # linecolor='k',
                cmap="viridis",
                annot=True,
                annot_kws={'fontsize':'large'},
                xticklabels=[language_labels[x] for x in LANGUAGES],
                yticklabels=[language_labels[x] for x in LANGUAGES] + ['Multi language'])
    plt.yticks(rotation=0)
    ax.set(ylabel="Model language", xlabel='Validation set ' + validation_set_no + ' language', title='ROC-AUC')

    plt.tight_layout()
    plt.savefig(OUT_PATH + 'multi_language_project_roc_auc_matrix_VS' + validation_set_no + '.png')


def roc_auc_boxplots(validation_set_no):
    multi_df = pandas.read_csv(OUT_PATH + 'artifact_detection_multi_language_model_resample_summary.csv')

    results_df = pandas.DataFrame()
    multi_boxes = []
    lang_boxes = []
    for lang in LANGUAGES:
        lang_df = pandas.read_csv(CROSS_LANGUAGE_EVALUATION + lang + '_artifact_detection_cross_language_resample_summary.csv')
        mult_res = evaluate_bootstrap(multi_df['roc-auc_' + lang + '_researcher_' + validation_set_no], 'Multilang' + '_VS' + validation_set_no)
        lang_res = evaluate_bootstrap(lang_df['roc-auc_' + lang + '_researcher_' + validation_set_no], lang + '_VS' + validation_set_no)
        results_df = results_df.append(pandas.DataFrame([mult_res]))
        results_df = results_df.append(pandas.DataFrame([lang_res]))
        multi_boxes.append(get_box(mult_res))
        lang_boxes.append(get_box(lang_res))

    results_df.to_csv(OUT_PATH + 'multi_language_roc_auc_bootstrap_VS' + validation_set_no + '.csv')
    fig, ax1 = plt.subplots(figsize=(8, 4))

    space = 0.2 # boxprops=dict(facecolor='tab:blue'),
    boxplot1 = ax1.bxp(multi_boxes, showfliers=False, widths=0.4, patch_artist=True, boxprops=dict(facecolor='lightcyan'), medianprops=dict(color="black", linewidth=1.5), positions=np.arange(5)-space,)
    ax2 = ax1.twinx()
    boxplot2 = ax2.bxp(lang_boxes, showfliers=False, widths=0.4, patch_artist=True, boxprops=dict(facecolor='lightgreen'), medianprops=dict(color="black", linewidth=1.5), positions=np.arange(5)+space,)

    ax1_lim = ax1.get_ylim()
    ax2_lim = ax2.get_ylim()

    # ax1.set_ylim(min(ax1_lim[0], ax2_lim[0]), max(ax1_lim[1], ax2_lim[1]))
    ax1.set_ylim(0.88, 0.97)
    ax2.set_ylim(0.88, 0.97)
    ax2.set_yticks([])

    ax1.set_xticks(np.arange(5))
    ax1.set_xticklabels([f'{label}' for label in language_labels.values()])

    ax1.set_ylabel('ROC-AUC')
    ax1.set_title('')
    plt.sca(ax1)
    plt.xticks(rotation=45)


    plt.legend(handles=[mpatches.Patch(color='lightcyan', label='Multi language model'),
                        mpatches.Patch(color='lightgreen', label='Language specific model')],
               loc='lower left')


    plt.tight_layout()
    # plt.show()
    plt.savefig(OUT_PATH + 'multi_language_roc_auc_boxplots_VS' + validation_set_no + '.png')


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

def plot_bootstrap_boxdiagram(fig, ax, title, metric, bootstrap_results_df, widths=None):
    boxes = []
    colors = []
    for index, row in bootstrap_results_df.sort_values('label').iterrows():
        box = {
            'label': row['label'],
            'whislo': row['lower']/100,
            'q1': row['lower']/100,
            'med': row['mean'],
            'q3': row['upper']/100,
            'whishi': row['upper']/100,
            'fliers': []
        }
        if row['label'].lower().startswith('ens') or row['label'].lower().startswith(' weighted'):
            colors.append('lightblue')
        else:
            colors.append('white')
        boxes.append(box)

    boxplot = ax.bxp(boxes, showfliers=False, patch_artist=True, medianprops=dict(color="black", linewidth=1.5), widths=widths)

    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel(metric)
    ax.set_title(title)
    plt.sca(ax)
    plt.xticks(rotation=45)


# def plot_numpy_confusion_matrix(cm, target_names):
#     disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=target_names)
#     disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal', values_format=None)
#     return disp


if __name__ == "__main__":
    main()
