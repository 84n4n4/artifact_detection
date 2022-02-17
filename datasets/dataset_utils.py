from os.path import dirname

import pandas

from artifact_detection_model.constants import TARGET_NAMES
from artifact_detection_model.regex_cleanup import split_by_md_code_block, regex_cleanup
from datasets.constants import LANGUAGES
from file_anchor import root_dir

def get_all_validation_sets():
    validation_sets = {}
    for lang in LANGUAGES:
        r_1_val_set = pandas.read_csv(root_dir() + 'datasets/' + lang + '_reseracher_1_manually_labeled_validation_set.csv.zip', compression='zip')
        r_2_val_set = pandas.read_csv(root_dir() + 'datasets/' + lang + '_reseracher_2_manually_labeled_validation_set.csv.zip', compression='zip')
        validation_sets.update({lang + '_researcher_1': r_1_val_set,
                                lang + '_researcher_2': r_2_val_set})
    return validation_sets


def get_data_from_issues(df, regex_clean=True):
    print(df.shape)
    df = df[df['body'].str.contains("```", na=False)]
    print(df.shape)
    docs = df['title'] + '\n' + df['body']
    documents = docs.tolist()

    artifacts, text = split_by_md_code_block(documents)

    if regex_clean:
        art, text = regex_cleanup(text)
        artifacts.extend(art)

    return artifacts, text


def get_data_from_documentation(df, regex_clean=True):
    df = df[~df['doc'].isnull()]
    df['doc'] = df['doc'].astype(str)
    documents = df.pop('doc').values

    artifacts, text = split_by_md_code_block(documents)

    if regex_clean:
        art, text = regex_cleanup(text)
        artifacts.extend(art)

    return artifacts, text


def get_trainingset(lang):
    df = pandas.read_csv(root_dir() + 'datasets/' + lang + '_training_issues.csv.zip', compression='zip')

    # todo add documentation
    artifacts, nat_lang = get_data_from_issues(df)

    df_nat_lang = pandas.DataFrame({'doc': nat_lang})
    df_nat_lang['target'] = TARGET_NAMES['text']
    df_artifacts = pandas.DataFrame({'doc': artifacts})
    df_artifacts['target'] = TARGET_NAMES['artifact']
    df_train = df_nat_lang.append(df_artifacts.sample(len(df_nat_lang), random_state=42))
    # df_train = df_nat_lang.append(df_artifacts)
    return df_train