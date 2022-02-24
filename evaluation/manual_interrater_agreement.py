import pandas
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, cohen_kappa_score, f1_score, \
    accuracy_score

from artifact_detection_model.utils.Logger import Logger

from datasets.constants import LANGUAGES
from datasets.dataset_utils import get_all_validation_sets
from evaluation.krippendorff import krippendorff
from evaluation.utils import plot_numpy_confusion_matrix
from file_anchor import root_dir

log = Logger()

out_path = root_dir() + 'evaluation/out/interrater_agreement/'
target_names = {'artifact': 0,
                'text': 1}


def combine_data_sets(reviewer1_df, reviewer2_df):
    reviewer1_df.rename(columns={'doc': 'doc1', 'target': 'target1'}, inplace=True)
    reviewer2_df.rename(columns={'doc': 'doc2', 'target': 'target2'}, inplace=True)
    combined = pandas.concat([reviewer1_df, reviewer2_df], axis=1)
    combined = combined[~combined['target1'].isnull()]
    combined = combined[~combined['target2'].isnull()]
    combined = combined[~combined['doc1'].isnull()]
    combined = combined[~combined['doc2'].isnull()]
    combined['target1'] = combined['target1'].astype('float')
    combined['target2'] = combined['target2'].astype('float')
    return combined


def main():
    val_sets = get_all_validation_sets()
    for lang in LANGUAGES:
        r_1_df = val_sets[lang + '_researcher_1']
        r_2_df = val_sets[lang + '_researcher_2']
        interrater_agreement_per_language(lang, r_1_df, r_2_df)


def interrater_agreement_per_language(language, r_1_df, r_2df):

    all_df = combine_data_sets(r_1_df, r_2df)
    all_df[all_df['target1'] != all_df['target2']].to_csv(out_path + language + '_reviewer1_vs_reviewer2_mismatched.csv')

    print(all_df['target1'].value_counts())
    print(all_df['target2'].value_counts())
    r1_target = all_df['target1'].to_list()
    r2_target = all_df['target2'].to_list()

    cm = confusion_matrix(r1_target, r2_target)
    disp = plot_numpy_confusion_matrix(cm, list(target_names.keys()))
    disp.ax_.set_ylabel('Researcher 1')
    disp.ax_.set_xlabel('Researcher 2')
    plt.gcf().subplots_adjust(left=0.2)
    plt.tight_layout()
    plt.savefig(out_path + language + '_confusion_matrix_reviewer1_vs_reviewer2.png')

    ir_metrics = [{'cohens_kappa': cohen_kappa_score(r1_target, r2_target),
                   'weighted_f1': f1_score(r1_target, r2_target, average='weighted'),
                   'macro_f1': f1_score(r1_target, r2_target, average='macro'),
                   'accuracy': accuracy_score(r1_target, r2_target),
                   'krippendorff_alpha': krippendorff.alpha([r1_target, r2_target]),
                   'roc_auc': roc_auc_score(r1_target, r2_target)}]
    pandas.DataFrame(ir_metrics).to_csv(out_path + language + '_reviewer1_vs_reviewer2_manual_agreement.csv')

    print('cohen ' + str(cohen_kappa_score(r1_target, r2_target)))
    print('f1 researcher 1 base ' + str(f1_score(r1_target, r2_target, average='weighted')))
    print('f1 researcher 2 base ' + str(f1_score(r2_target, r1_target, average='weighted')))
    print('accuracy ' + str(accuracy_score(r1_target, r2_target)))
    print('krippendorff alpha ' + str(krippendorff.alpha([r1_target, r2_target])))
    print('roc auc Reviewer1 base ' + str(roc_auc_score(r1_target, r2_target)))
    print('roc auc Reviewer2 base ' + str(roc_auc_score(r2_target, r1_target)))


if __name__ == "__main__":
    main()
