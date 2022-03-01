import random
import traceback

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

OUT_PATH = root_dir() + 'evaluation/out/learning_curve/no_case_folding_keep_cases/'


def main1():
    fig = plt.figure(figsize=(8, 8))
    axes = []
    axes.append(fig.add_subplot(3, 2, 1))
    axes.append(fig.add_subplot(3, 2, 2, sharex=axes[0], sharey=axes[0]))
    axes.append(fig.add_subplot(3, 2, 3, sharex=axes[0], sharey=axes[0]))
    axes.append(fig.add_subplot(3, 2, 4, sharex=axes[0], sharey=axes[0]))
    axes.append(fig.add_subplot(3, 2, 5, sharex=axes[0], sharey=axes[0]))
    axes.append(fig.add_subplot(3, 2, 6, sharex=axes[0]))

    # fig, axes = plt.subplots(3, 2, figsize=(8, 8), sharex=True, sharey=True)

    for i, lang in enumerate(LANGUAGES):
        print(lang)
        df = pandas.read_csv(OUT_PATH + lang + '_artifact_detection_summary.csv')
        df = df[df['train_samples'] < 800000]
        df['train_samples'] = df['train_samples']/ 10**3
        ax = axes[int(i/2)][i % 2]
        gb = df.groupby(by='train_samples')

        plot_mean_and_fill_std(ax, gb, 'roc-auc_' + lang + '_researcher_1', 'g', 'Validation set 1')
        plot_mean_and_fill_std(ax, gb, 'roc-auc_' + lang + '_researcher_2', 'b', 'Validation set 2')
        # plt.axvline(x=288038, color='gray', label='40% Training set')
        ax.title.set_text(lang)
        if not i % 2:
            ax.set_ylabel('ROC-AUC')
        if i > 3:
            ax.set_xlabel('Training set size (1000 lines)')
        if i == 0:
            ax.legend(loc='lower right')

        colors = ['r', 'g', 'b', 'k', 'c']
        plot_mean_and_fill_std(axes[2][1], gb, 'model_size', colors[i], lang)
        axes[2][1].set_ylabel('Model size (MiB)')
        axes[2][1].set_xlabel('Training set size')

    plt.tight_layout()

    plt.show()

    # plt.savefig(OUT_PATH + language + '_roc-auc_validation_set_learning_curve.png')




def scoring_report(df, lang):
    df = df[df['train_frac'] == 25000]
    df.mean().to_csv(OUT_PATH + lang + '_means.csv')


def plot_learning_curve(df, language):
    # df = df[df['train_frac'] > 0.1]
    # df[['index', 'train_frac', 'train_samples', 'macro_f1', 'macro_f1_reviewer_2', 'roc-auc', 'roc-auc_reviewer_2']]
    gb = df.groupby(by='train_samples')

    # validation set
    fig, axes = plt.subplots(1, 1, figsize=(5, 4))
    plot_mean_and_fill_std(axes, gb, 'macro_f1_' + language + '_researcher_1', 'g', 'Validation set 1')
    plot_mean_and_fill_std(axes, gb, 'macro_f1_' + language + '_researcher_2', 'b', 'Validation set 2')
    # plt.axvline(x=288038, color='gray', label='40% Training set')
    axes.set_ylabel('F1')
    axes.set_xlabel('Training set size')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PATH + language + '_macro_f1_validation_set_learning_curve.png')
    plt.close()

    fig, axes = plt.subplots(1, 1, figsize=(5, 4))
    plot_mean_and_fill_std(axes, gb, 'roc-auc_' + language + '_researcher_1', 'g', 'Validation set 1')
    plot_mean_and_fill_std(axes, gb, 'roc-auc_' + language + '_researcher_2', 'b', 'Validation set 2')
    # plt.axvline(x=288038, color='gray', label='40% Training set')
    axes.set_ylabel('ROC-AUC')
    axes.set_xlabel('Training set size')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(OUT_PATH + language + '_roc-auc_validation_set_learning_curve.png')
    plt.close()

    # # nlon
    # fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    # plot_mean_and_fill_std(axes, gb, 'roc-auc', 'r', 'Test set')
    # plot_mean_and_fill_std(axes, gb, 'roc-auc_nlon_all_Fabio', 'b', 'NLoN Fabio')
    # plot_mean_and_fill_std(axes, gb, 'roc-auc_nlon_all_Mika', 'g', 'NLoN Mika')
    # axes.set_ylabel('ROC-AUC')
    # axes.set_xlabel('Training set size')
    # plt.axvline(x=288038, color='gray', label='40% Training set')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(OUT_PATH + 'roc-auc_nlon_all_set_learning_curve.png')
    #
    # fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    # plot_mean_and_fill_std(axes, gb, 'macro_f1', 'r', 'Test set')
    # plot_mean_and_fill_std(axes, gb, 'macro_f1_nlon_all_Fabio', 'b', 'NLoN Fabio')
    # plot_mean_and_fill_std(axes, gb, 'macro_f1_nlon_all_Mika', 'g', 'NLoN Mika')
    # axes.set_ylabel('F1')
    # axes.set_xlabel('Training set size')
    # plt.axvline(x=288038, color='gray', label='40% Training set')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(OUT_PATH + 'macro_f1_nlon_all_set_learning_curve.png')

    # model size
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    plot_mean_and_fill_std(axes, gb, 'model_size', 'r', 'Model size (MiB)')
    axes.set_ylabel('MiB')
    axes.set_xlabel('Training set size')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PATH + language + '_model_size_learning_curve.png')
    plt.close()

    # runtime performance
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    plot_mean_and_fill_std(axes, gb, 'perf_train_runtime', 'r', 'Training time')
    axes.set_ylabel('Seconds')
    axes.set_xlabel('Training set size')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PATH + language + '_perf_train_runtime_learning_curve.png')
    plt.close()

    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    plot_mean_and_fill_std(axes, gb, 'perf_predict_runtime_' + language + '_researcher_1', 'r', 'Validation set 1 classification time (perf_counter)')
    axes.set_ylabel('Seconds')
    axes.set_xlabel('Training set size')
    # plt.axvline(x=288038, color='gray', label='40% Training set')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PATH + language + '_perf_predict_runtime_learning_curve.png')
    plt.close()

    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    plot_mean_and_fill_std(axes, gb, 'timeit_runtime_' + language + '_researcher_1', 'r', 'Validation set 1 classification time (timeit)')
    axes.set_ylabel('Seconds')
    axes.set_xlabel('Training set size')
    # plt.axvline(x=288038, color='gray', label='40% Training set')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PATH + language + '_perf_timeit_predict_runtime_learning_curve.png')
    plt.close()


def plot_mean_and_fill_std(axes, gb, metric, color, label):
    axes.fill_between(gb.mean().index, gb.mean()[metric] - gb.std()[metric],
                         gb.mean()[metric] + gb.std()[metric], alpha=0.1,
                         color=color)
    axes.plot(gb.mean().index, gb.mean()[metric], 'o-', color=color, label=label)


def get_learning_curve_data(lang):
    df_train = get_trainingset(lang, balance=False)
    val_sets = get_validation_sets_for_language(lang)

    df = pandas.DataFrame()

    try:
        for train_frac in [6250, 12500, 25000, 50000, 100000, 200000, 400000, 800000, 1600000, 3200000]:
            if train_frac > len(df_train):
                act_train_frac = len(df_train)
            else:
                act_train_frac = train_frac
            for index in range(0, 10):
                seed = random.randint(100, 1000)

                df_sel = df_train[df_train['target'] == 1].sample(int(act_train_frac / 2), random_state=seed, replace=True)
                df_sel = df_sel.append( df_train[df_train['target'] == 0].sample(int(act_train_frac / 2), random_state=seed, replace=True))

                report, pipeline = run_ml_artifact_training(df_sel, LinearSVC(random_state=42))
                report.update({'seed': seed})
                report.update({'train_frac': act_train_frac})
                report.update({'index': index})

                for val_set_name, val_set_df in val_sets.items():
                    val_docs = val_set_df.copy().pop('doc').values
                    val_targets = val_set_df.copy().pop('target').values
                    report.update(validation_performance_on_dataset(pipeline, val_docs, val_targets, val_set_name))
                print(report)

                df = df.append(pandas.DataFrame([report]))
            if train_frac > len(df_train):
                break
    except Exception as e:
        log.e(str(e))
        log.e(str(traceback.format_tb(e.__traceback__)))

    df.to_csv(OUT_PATH + lang + '_artifact_detection_summary.csv')
    return df


def get_learning_curve_data_fractional(lang):
    df_train = get_trainingset(lang)
    val_sets = get_validation_sets_for_language(lang)

    df = pandas.DataFrame()

    try:
        for train_frac in [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1]:
            for index in range(0, 10):
                seed = random.randint(100, 1000)
                report, pipeline = run_ml_artifact_training(df_train.copy().sample(frac=train_frac, random_state=seed),
                                                            LinearSVC(random_state=42))
                report.update({'seed': seed})
                report.update({'train_frac': train_frac})
                report.update({'index': index})

                for val_set_name, val_set_df in val_sets.items():
                    val_docs = val_set_df.copy().pop('doc').values
                    val_targets = val_set_df.copy().pop('target').values
                    report.update(validation_performance_on_dataset(pipeline, val_docs, val_targets, val_set_name))
                print(report)

                df = df.append(pandas.DataFrame([report]))
    except Exception as e:
        log.e(str(e))
        log.e(str(traceback.format_tb(e.__traceback__)))
    df.to_csv(OUT_PATH + lang + '_artifact_detection_summary_fractional.csv')
    return df


if __name__ == "__main__":
    main()
