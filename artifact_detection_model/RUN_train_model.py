import json

import joblib
from sklearn.svm import LinearSVC

from artifact_detection_model.constants import TARGET_NAMES
from artifact_detection_model.model_training import run_ml_artifact_training
from artifact_detection_model.utils.Logger import Logger
from datasets.dataset_utils import get_trainingset, get_all_validation_sets
from evaluation.utils import validation_performance_on_dataset
from file_anchor import root_dir

log = Logger()

OUT_PATH = root_dir() + 'artifact_detection_model/out/'


def main():
    lang = 'cpp'
    seed = 42
    df_train = get_trainingset(lang)
    val_sets = get_all_validation_sets()
    train_frac = 0.4

    df_train = df_train.copy().sample(frac=train_frac, random_state=seed)
    df_train[df_train['target'] == 0]['doc'].to_csv(OUT_PATH + 'train_artifact.csv')
    df_train[df_train['target'] == 1]['doc'].to_csv(OUT_PATH + 'train_natural_lang.csv')

    report, pipeline = run_ml_artifact_training(df_train,
                                                LinearSVC(random_state=42))

    report.update({'seed': seed})
    report.update({'train_frac': train_frac})

    # df_train, df_test = get_training_and_test_set()
    # df_train = df_train.sample(frac=train_frac, random_state=42)
    # report, pipeline = run_ml_artifact_training(df_train, df_test, LinearSVC(random_state=42), model_output_path=OUT_PATH)

    report.update({'name': 'LSVCdef'})
    report.update({'train_frac': train_frac})

    for val_set_name, val_set_df in val_sets.items():
        val_docs = val_set_df.copy().pop('doc').values
        val_targets = val_set_df.copy().pop('target').values
        report.update(validation_performance_on_dataset(pipeline, val_docs, val_targets, val_set_name))

    with open(OUT_PATH + 'performance_report.json', 'w') as fd:
        json.dump(report, fd, indent=2)

    investigate_miscalssifications(pipeline, val_sets['cpp_researcher_1'], 'cpp_researcher_1')
    return report, pipeline


def investigate_miscalssifications(pipeline, val_set_df, val_set_name):
    data = val_set_df.copy().pop('doc').values
    target = val_set_df.copy().pop('target').values
    name = val_set_name

    y_predicted = pipeline.predict(data)

    wrongly_identified_as_artifact = []
    wrongly_identified_as_text = []
    for index in range(0, len(data)):
        if target[index] == y_predicted[index]:
            pass
        elif target[index] == TARGET_NAMES['artifact'] and y_predicted[index] == TARGET_NAMES['text']:
            wrongly_identified_as_text.append(data[index])
        else:
            wrongly_identified_as_artifact.append(data[index])

    with open(OUT_PATH + name + '_wrongly_identified_as_artifact.txt', 'w') as fd:
        fd.write('\n\n'.join(wrongly_identified_as_artifact))
    with open(OUT_PATH + name + '_wrongly_identified_as_text.txt', 'w') as fd:
        fd.write('\n\n'.join(wrongly_identified_as_text))


def store_model(pipeline):
    joblib.dump(pipeline, OUT_PATH + 'artifact_detection.joblib')


if __name__ == "__main__":
    main()
