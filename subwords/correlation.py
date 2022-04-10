import de_core_news_sm
import pandas as pd
from nltk.corpus import stopwords
from pkg_resources import resource_filename
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from tqdm import tqdm
from transformers import BertTokenizer

from ml_classifier.svc import get_config

bert_model = BertTokenizer.from_pretrained('bert-base-uncased')


def subwords_wo_stop():
    config = get_config('/../config/config.yaml')
    feature_neg_file_path = resource_filename(__name__, config['feature_neg_file_path']['path'])
    feature_pos_file_path = resource_filename(__name__, config['feature_pos_file_path']['path'])
    feature_neg_test_file_path = resource_filename(__name__, config['feature_neg_test_file_path']['path'])
    feature_pos_test_file_path = resource_filename(__name__, config['feature_pos_test_file_path']['path'])
    german_stop_words = stopwords.words('german')
    sp = de_core_news_sm.load()

    feature = pd.DataFrame(columns=['ADV', 'ADJ', 'NONE', 'PROPN', 'VERB'])
    length = []
    ls_split = []
    df = pd.read_csv(feature_neg_file_path, sep=';')
    neg_df = df.drop("Unnamed: 0", axis=1)
    pos_df = pd.read_csv(feature_pos_file_path, sep=',')
    with tqdm(total=neg_df.shape[0]) as pbar:
        for index, row in neg_df.iterrows():
            pbar.update(1)
            neg_text_input = row['test_input']
            tokens = bert_model.tokenize(neg_text_input)
            text_wo_stop_words = [word for word in tokens if word not in german_stop_words]  # removing stop words
            neg_l = len(text_wo_stop_words)
            length.append(neg_l)
            neg_split = 0
            for i in text_wo_stop_words:
                if "##" in i:
                    neg_split += 1
            ls_split.append(neg_split)
            sentence = ' '.join(text_wo_stop_words)
            sen = sp(sentence)
            VERB = 0
            PROPN = 0
            NONE = 0
            ADV = 0
            ADJ = 0
            occurance = []
            for token in sen:
                pos = token.pos_
                if pos == "ADV":
                    ADV += 1
                elif pos == "ADJ":
                    ADJ += 1
                elif pos == "NONE":
                    NONE += 1
                elif pos == "PROPN":
                    PROPN += 1
                elif pos == "VERB":
                    VERB += 1
            occurance.append(ADV)
            occurance.append(ADJ)
            occurance.append(NONE)
            occurance.append(PROPN)
            occurance.append(VERB)
            s = pd.Series(occurance, index=feature.columns)
            feature = feature.append(s, ignore_index=True)
        neg_index = len(feature.index)

    with tqdm(total=pos_df.shape[0]) as pbar:
        for index, row in pos_df.iterrows():
            pbar.update(1)
            pos_text_input = row['text']
            tokens = bert_model.tokenize(pos_text_input)
            text_wo_stop_words = [word for word in tokens if word not in german_stop_words]  # removing stop words
            pos_l = len(text_wo_stop_words)
            length.append(pos_l)
            pos_split = 0
            for i in text_wo_stop_words:
                if "##" in i:
                    pos_split += 1
            ls_split.append(pos_split)
            sentence = ' '.join(text_wo_stop_words)
            sen = sp(sentence)
            VERB = 0
            PROPN = 0
            NONE = 0
            ADV = 0
            ADJ = 0
            occurance = []
            for token in sen:
                pos = token.pos_
                if pos == "ADV":
                    ADV += 1
                elif pos == "ADJ":
                    ADJ += 1
                elif pos == "NONE":
                    NONE += 1
                elif pos == "PROPN":
                    PROPN += 1
                elif pos == "VERB":
                    VERB += 1
            occurance.append(ADV)
            occurance.append(ADJ)
            occurance.append(NONE)
            occurance.append(PROPN)
            occurance.append(VERB)
            s = pd.Series(occurance, index=feature.columns)
            feature = feature.append(s, ignore_index=True)
        feature['length'] = length
        feature['split'] = ls_split
        feature['sentiment'] = [0] * neg_index + [1] * (len(feature.index) - neg_index)

    print(feature[feature.columns[1:]].corr()['sentiment'][:])

    correlations = cdist(feature.iloc[:, :-1].to_numpy().transpose(), feature['sentiment'].to_numpy().reshape(1, -1),
                         metric='correlation')
    print(correlations)

    print(distance.correlation(feature['split'].to_numpy(),
                               feature['sentiment'].to_numpy()))
    print(distance.correlation(feature['length'].to_numpy(),
                               feature['sentiment'].to_numpy()))
    print(distance.correlation(feature['ADV'].to_numpy(),
                               feature['sentiment'].to_numpy()))
    print(distance.correlation(feature['ADJ'].to_numpy(),
                               feature['sentiment'].to_numpy()))
    print(distance.correlation(feature['NONE'].to_numpy(),
                               feature['sentiment'].to_numpy()))
    print(distance.correlation(feature['PROPN'].to_numpy(),
                               feature['sentiment'].to_numpy()))
    print(distance.correlation(feature['VERB'].to_numpy(),
                               feature['sentiment'].to_numpy()))


if __name__ == "__main__":
    subwords_wo_stop()
