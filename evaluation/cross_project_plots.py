import random
import traceback

import numpy as np
import pandas
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import LinearSVC

from artifact_detection_model.model_training import run_ml_artifact_training
from artifact_detection_model.utils.Logger import Logger
from datasets.constants import LANGUAGES
from datasets.dataset_utils import get_trainingset, get_all_validation_sets, get_validation_sets_for_language
from evaluation.stats_utils import evaluate_bootstrap, t_test_x_greater_y
from evaluation.utils import validation_performance_on_dataset
from file_anchor import root_dir

log = Logger()

OUT_PATH = root_dir() + 'evaluation/out/cross_language/'

language_labels = {
    'cpp': 'C++',
    'java': 'Java',
    'javascript': 'JavaScript',
    'php': 'PHP',
    'python': 'Python',
}

# man_validation_samples_cpp_researcher_1
# classification_report_cpp_researcher_1
# macro_f1_cpp_researcher_1
# roc-auc_cpp_researcher_1
# perf_predict_runtime_cpp_researcher_1
# timeit_runtime_cpp_researcher_1


RESEARCHER = '1'


def main():
    cross_project_roc_auc_matrix()
    p_test_model_trained_performs_better_on_its_own_language_than_other_languages()


def p_test_model_trained_performs_better_on_its_own_language_than_other_languages():
    rep_df = pandas.DataFrame()
    for lang in LANGUAGES:
        df = pandas.read_csv(OUT_PATH + lang + '_artifact_detection_cross_language_resample_summary.csv')

        for l in LANGUAGES:
            if lang == l:
                continue
            rep = t_test_x_greater_y(df['roc-auc_' + lang + '_researcher_' + RESEARCHER],
                                     df['roc-auc_' + l + '_researcher_' + RESEARCHER],
                                     lang, l)  # one sided, x greater y
            rep['model'] = lang
            rep_df = rep_df.append(rep)

    rep_df.to_csv(OUT_PATH + 'cross_project_model_better_on_its_own_language_than_other_languages.csv')


def cross_project_roc_auc_matrix():
    columns = ['roc-auc_' + x + '_researcher_' + RESEARCHER for x in LANGUAGES]
    cm = []
    for lang in LANGUAGES:
        df = pandas.read_csv(OUT_PATH + lang + '_artifact_detection_cross_language_resample_summary.csv')
        df = df[columns].mean()
        cm.append(df.to_list())
    disp = plot_numpy_confusion_matrix(cm, [language_labels[x] for x in LANGUAGES])
    disp.ax_.set(ylabel="Model language", xlabel="Validation set language", title='ROC-AUC')
    plt.savefig(OUT_PATH + 'cross_project_roc_auc_matrix.png')
    # plt.show()


def plot_numpy_confusion_matrix(cm, target_names):
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=target_names)
    disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal', values_format=None)
    return disp


if __name__ == "__main__":
    main()
