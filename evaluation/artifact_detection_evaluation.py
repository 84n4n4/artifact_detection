import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.svm import LinearSVC
from sklearn.utils import resample

from artifact_detection_model.constants import TARGET_NAMES
from artifact_detection_model.dataset_creation import get_nlon_dataset, get_data_from_issues
from artifact_detection_model.model_training import run_ml_artifact_training
from artifact_detection_model.utils.Logger import Logger
from file_anchor import root_dir

log = Logger()

OUT_PATH = root_dir() + "evaluation/out/evaluation/"

balance = True
reviewer = 'Fabio'


def get_dataset(lang):
    df = pandas.read_csv(root_dir() + lang + '_training_issues.csv.zip', compression='zip')
    get_data_from_issues(df)

    artifacts, nat_lang = get_data_from_issues(df)

    df_nat_lang = pandas.DataFrame({'doc': nat_lang})
    df_nat_lang['target'] = TARGET_NAMES['text']
    df_artifacts = pandas.DataFrame({'doc': artifacts})
    df_artifacts['target'] = TARGET_NAMES['artifact']
    df_train = df_nat_lang.append(df_artifacts.sample(len(df_nat_lang), random_state=42))
    # df_train = df_nat_lang.append(df_artifacts)
    return df_train

def main():
    lang = 'java'

    df = get_dataset(lang)
    docs = df.copy().pop('doc').values
    target = df.copy().pop('target').values

    n_iterations = 10
    # n_iterations = 100
    n_size = int(len(docs)*0.8)

    df = pandas.DataFrame()
    for i in range(n_iterations):
        # prepare train and test sets
        docs_indices = list(range(0, len(docs)))
        train_idx, t_ = resample(docs_indices, target, n_samples=n_size, stratify=target)
        train_x = docs[train_idx]
        train_y = target[train_idx]

        test_idx = [x for x in docs_indices if x not in list(train_idx)]
        test_x = docs[test_idx]
        test_y = target[test_idx]

        df_train = pandas.DataFrame({'doc': train_x, 'target': train_y})
        df_test = pandas.DataFrame({'doc': test_x, 'target': test_y})

        # fit model
        report, _ = run_ml_artifact_training(df_train, df_test, LinearSVC(random_state=42))
        df = df.append(pandas.DataFrame([report]))

    df.to_csv(OUT_PATH + 'reports.csv')
    evaluate_bootstrap(df, 'macro_f1')
    evaluate_bootstrap(df, 'roc-auc')


def evaluate_bootstrap(df, metric):
    df[metric].plot(kind='hist')
    plt.savefig(OUT_PATH + metric + '.png')
    plt.close()

    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(df[metric], p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(df[metric], p))
    mean = df[metric].mean()
    pandas.DataFrame([{'alpha': alpha*100, 'lower': lower*100, 'upper': upper*100, 'mean': mean}]).to_csv(OUT_PATH + metric + '.csv')
    print(metric + ': %.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))


if __name__ == "__main__":
    main()
