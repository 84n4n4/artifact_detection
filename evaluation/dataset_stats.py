import random

import pandas
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from artifact_detection_model.model_training import run_ml_artifact_training
from artifact_detection_model.utils.Logger import Logger
from datasets.constants import LANGUAGES
from datasets.dataset_utils import get_trainingset, get_all_validation_sets, get_validation_sets_for_language
from evaluation.utils import validation_performance_on_dataset
from file_anchor import root_dir

log = Logger()

OUT_PATH = root_dir() + 'evaluation/out/dataset_stats/'


def main():
    reports = []
    for lang in LANGUAGES:
        df_train = get_trainingset(lang)
        val_sets = get_validation_sets_for_language(lang)
        report = {'len_df_train': len(df_train)}

        for val_set_name, val_set_df in val_sets.items():
            report.update({
                'text_' + val_set_name: len(val_set_df[val_set_df['target'] == 1]),
                'artifact_' + val_set_name: len(val_set_df[val_set_df['target'] == 0]),
                'total_' + val_set_name: len(val_set_df)
            })
        reports.append(report)
    pandas.DataFrame(reports).to_csv(OUT_PATH + 'dataset_stats.csv')


if __name__ == "__main__":
    main()
