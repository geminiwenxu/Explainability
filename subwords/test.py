import de_core_news_sm
import pandas as pd
from nltk.corpus import stopwords
from pkg_resources import resource_filename
from scipy.spatial.distance import cdist
from transformers import BertTokenizer

from ml_classifier.svc import get_config


def subwords_wo_stop():
    config = get_config('/../config/config.yaml')
    feature_neg_file_path = resource_filename(__name__, config['feature_neg_file_path']['path'])
    feature_pos_file_path = resource_filename(__name__, config['feature_pos_file_path']['path'])
    feature_test_file_path = resource_filename(__name__, config['feature_test_file_path']['path'])
    german_stop_words = stopwords.words('german')
    sp = de_core_news_sm.load()

    feature = pd.DataFrame(columns=['ADV', 'ADJ', 'NONE', 'PROPN', 'VERB'])
    length = []
    ls_split = []
    df = pd.read_csv(feature_test_file_path, sep=';')
    result = df.drop("Unnamed: 0", axis=1)

    for index, row in result.iterrows():
        neg_text_input = row['test_input']
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokens = tokenizer.tokenize(neg_text_input)
        # print(neg_text_input)
        # print(tokens)
        # sen_split = re.findall(r'\w+|\S+', tokens)
        # low_sen_split = [i.lower() for i in sen_split]
        # print(low_sen_split)
        text_wo_stop_words = [word for word in tokens if word not in german_stop_words]  # removing stop words
        # print(text_wo_stop_words)
        l = len(text_wo_stop_words)
        length.append(l)
        split = 0
        for i in text_wo_stop_words:
            if "##" in i:
                split += 1
        ls_split.append(split)
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
        # print(occurance)
        s = pd.Series(occurance, index=feature.columns)
        feature = feature.append(s, ignore_index=True)
    feature['length'] = length
    feature['split'] = ls_split
    feature['sentiment'] = [0] * len(feature.index)

    print(feature[feature.columns[1:]].corr()['sentiment'][:])

    correlations = cdist(feature.iloc[:, :-1].to_numpy().transpose(), feature['sentiment'].to_numpy().reshape(1, -1),
                         metric='correlation')
    print(correlations)


if __name__ == "__main__":
    subwords_wo_stop()
