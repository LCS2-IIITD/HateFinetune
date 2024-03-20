BASE_PATH = '../../Saves/prepared_data/'

import pandas as pd
from sklearn.model_selection import train_test_split

def load_custom_ds(dset_name):
    if dset_name == 'olid_taska':
        df_train = pd.read_csv(BASE_PATH + 'olid_train_A.csv', index_col=False)
        df_test = pd.read_csv(BASE_PATH + 'olid_test_A.csv', index_col=False)
        df_train, df_validation = train_test_split(df_train, test_size=0.25, random_state=42)
        return df_train, df_validation, df_test
    elif dset_name == 'davidson':
        df_train = pd.read_csv(BASE_PATH + 'davidson_train.csv', index_col=False)
        df_test = pd.read_csv(BASE_PATH + 'davidson_test.csv', index_col=False)
        df_train.rename(columns={'final_posts': 'text',
                           'final_labels': 'label'},
                  inplace=True, errors='raise')
        df_test.rename(columns={'final_posts': 'text',
                           'final_labels': 'label'},
                  inplace=True, errors='raise')
        df_train, df_validation = train_test_split(df_train, test_size=0.25, random_state=42)
        return df_train, df_validation, df_test
    elif dset_name == 'dynabench_label':
        df_train = pd.read_csv(BASE_PATH + 'df_dynabench_label_detection_train.csv', index_col=False)
        df_test = pd.read_csv(BASE_PATH + 'df_dynabench_label_detection_test.csv', index_col=False)
        df_train, df_validation = train_test_split(df_train, test_size=0.25, random_state=42)
        return df_train, df_validation, df_test
    elif dset_name == "hatexplain_label":
        df_train = pd.read_csv(BASE_PATH + 'df_hateXplain_train_label_pred.csv', index_col=False)
        df_test = pd.read_csv(BASE_PATH + 'df_hateXplain_test_label_pred.csv', index_col=False)
        df_train.rename(columns={'final_posts': 'text',
                           'final_labels': 'label'},
                  inplace=True, errors='raise')
        df_test.rename(columns={'final_posts': 'text',
                           'final_labels': 'label'},
                  inplace=True, errors='raise')
        df_train, df_validation = train_test_split(df_train, test_size=0.25, random_state=42)
        return df_train, df_validation, df_test
    elif dset_name == 'waseem':
        df_train = pd.read_csv(BASE_PATH + 'df_waseem_train.csv', index_col=False)
        df_test = pd.read_csv(BASE_PATH + 'df_waseem_test.csv', index_col=False)
        df_train.rename(columns={'final_posts': 'text',
                           'final_labels': 'label'},
                  inplace=True, errors='raise')
        df_test.rename(columns={'final_posts': 'text',
                           'final_labels': 'label'},
                  inplace=True, errors='raise')
        df_train, df_validation = train_test_split(df_train, test_size=0.25, random_state=42)
        return df_train, df_validation, df_test
    elif dset_name == 'founta':
        df_train = pd.read_csv(BASE_PATH + 'founta_train.csv', index_col=False)
        df_test = pd.read_csv(BASE_PATH + 'founta_test.csv', index_col=False)
        df_train.rename(columns={'full_text': 'text',
                           'label': 'label'},
                  inplace=True, errors='raise')
        df_test.rename(columns={'full_text': 'text',
                           'label': 'label'},
                  inplace=True, errors='raise')
        df_train, df_validation = train_test_split(df_train, test_size=0.25, random_state=42)
        return df_train, df_validation, df_test
    elif dset_name == 'toxigen_label':
        df_train = pd.read_csv(BASE_PATH + 'toxigen_label_train.csv', index_col=False)
        df_test = pd.read_csv(BASE_PATH + 'toxigen_label_test.csv', index_col=False)
        df_train.rename(columns={'generation': 'text',
                           'prompt_label': 'label'},
                  inplace=True, errors='raise')
        df_test.rename(columns={'generation': 'text',
                           'prompt_label': 'label'},
                  inplace=True, errors='raise')
        replacement_dict = {0: 'NonHate', 1: 'Hate'}
        df_train['label'] = df_train['label'].replace(replacement_dict)
        df_test['label'] = df_test['label'].replace(replacement_dict)
        df_train, df_validation = train_test_split(df_train, test_size=0.25, random_state=42)
        return df_train, df_validation, df_test