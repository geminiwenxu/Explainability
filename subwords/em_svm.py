import numpy as np
import pandas as pd
from pkg_resources import resource_filename
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tqdm import tqdm
from transformers import BertTokenizer

from ml_classifier.svc import get_config
from subwords.embedding import embeddings

bert_model = BertTokenizer.from_pretrained('bert-base-uncased')
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def em_cluster():
    config = get_config('/../config/config.yaml')
    feature_neg_test_file_path = resource_filename(__name__, config['feature_neg_test_file_path']['path'])
    feature_pos_test_file_path = resource_filename(__name__, config['feature_pos_test_file_path']['path'])
    df = pd.read_csv(feature_neg_test_file_path, sep=';')
    neg_df = df.drop("Unnamed: 0", axis=1)
    pos_df = pd.read_csv(feature_pos_test_file_path, sep=',')

    neg_ls_sen = []
    with tqdm(total=neg_df.shape[0]) as pbar:
        for index, row in neg_df.iterrows():
            pbar.update(1)
            neg_text_input = row['test_input']
            tokens = bert_model.tokenize(neg_text_input)
            if len(tokens) < 510:
                try:
                    em = embeddings(neg_text_input)
                    neg_ls_sen.append(em)
                except ValueError:
                    print('value error')
    pos_ls_sen = []
    with tqdm(total=pos_df.shape[0]) as pbar:
        for index, row in pos_df.iterrows():
            pbar.update(1)
            pos_text_input = row['text']
            tokens = bert_model.tokenize(pos_text_input)
            if len(tokens) < 510:
                try:
                    em = embeddings(pos_text_input)
                    pos_ls_sen.append(em)
                except ValueError:
                    print('value error')
    ls = neg_ls_sen + pos_ls_sen
    X = np.asarray(ls)
    Y = [0] * len(neg_ls_sen) + [1] * len(pos_ls_sen)
    Y = np.asarray(Y)
    print(type(X))
    print(X.shape)
    print(Y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    clf = SVC(gamma='auto', cache_size=500)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    class_names = ['correct_classified', 'misclassified']
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    em_cluster()
