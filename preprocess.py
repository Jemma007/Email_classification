import pandas as pd
import numpy as np
import os
import random

SEED = 2022
random.seed(SEED)

def normalise_text(text):
    text = text.lower()
    text = text.strip()
    text = text.strip('\n')
    text = text.replace(r"\#", "")
    text = text.replace(r"http\S+", "URL")
    text = text.replace(r"@", "")
    text = text.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ")
    text = text.replace("\s{2,}", " ")
    text = text.replace(r'[_.!+-=——,$%^，。？、~@#￥%……&*《》<>「」{}【】()/|]', '')
    text = text.replace('\0', '')
    text = text.split()
    return ' '.join(text)

def split_train_test(dataset, ratio=0.7):
    idxs = list(range(len(dataset)))
    random.shuffle(idxs)
    train_idxs = idxs[:int(len(dataset)*ratio)]
    test_idxs = idxs[int(len(dataset)*ratio):]
    train_data = dataset.iloc[train_idxs, :].copy()
    test_data = dataset.iloc[test_idxs, :].copy()
    return train_data, test_data


def tras2csv(root):
    labels = os.listdir(root)
    dataset = []

    for i, label in enumerate(labels):
        data_root = os.path.join(root, label)
        files_name = os.listdir(data_root)

        for file_name in files_name:
            file_path = os.path.join(data_root, file_name)
            curr_email = ''
            with open(file_path, mode='r', encoding='gbk', errors="ignore") as cur_file:
                for line in cur_file:
                    curr_email += normalise_text(line)
                    curr_email += ' '
            dataset.append({'text': curr_email.strip(), 'label': label})
    dataset = pd.DataFrame(dataset)
    # print(dataset.head())
    train_data, test_data = split_train_test(dataset)
    train_data = train_data.reset_index()
    train_data = train_data.drop('index', axis=1)
    train_data, valid_data = split_train_test(train_data)
    train_data.to_csv('./data/train.csv', index=False)
    valid_data.to_csv('./data/valid.csv', index=False)
    test_data.to_csv('./data/test.csv', index=False)

if __name__ == '__main__':
    tras2csv('./data')
