from pkg_resources import resource_filename
import yaml
import json
import pandas as pd
from feature.feature_generator import Scorer
from feature.feature_generator import ADJPD, AdjustedModulus, ADVPD, Alpha, APD, ATL, AutoBERTT, ASL
from feature.feature_generator import CurveLength, DPD, Entropy, Gini, HL, HPoint, IPD, NPD, Lambda, lmbd, NDW
from feature.feature_generator import PPD, PREPPD, Q, R1, RR, RRR, STC, Syn, TC, TypeTokenRatio, uniquegrams, VD, VPD
from transformers import BertTokenizer

print("Start")


def get_config(path: str) -> dict:
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def main():
    config = get_config('/../config/config.yaml')
    feature_pos_file_path = resource_filename(__name__, config['feature_pos_file_path']['path'])
    feature_test_file_path = resource_filename(__name__, config['feature_test_file_path']['path'])

    df = pd.read_csv(feature_pos_file_path, sep=',')
    data = []
    for index, row in df.iterrows():
        dict = {}
        text_input = row['text']
        tokenizer = BertTokenizer.from_pretrained(config['pre_trained_model_name'])
        tokens = tokenizer.tokenize(text_input)
        print(text_input)
        # print(len(tokens))
        if len(tokens) < 512:
            try:
                sc = Scorer(scorers=[ADJPD(), AdjustedModulus(), ADVPD(), Alpha(), APD(), ATL(), AutoBERTT(), ASL(),
                                     CurveLength(), DPD(), Entropy(), Gini(), HL(), HPoint(), IPD(), NPD(), Lambda(),
                                     lmbd(), NDW(), PPD(), PREPPD(), Q(), R1(), RR(), RRR(), STC(), Syn(), TC(),
                                     TypeTokenRatio(), uniquegrams(), VD(), VPD()])
                scores, names, text_hash = sc.run("de", "dummy", text_input)
                # print(scores)
                # print(names)
                # print(text_hash)
                dict['text'] = text_input
                dict['feature'] = scores
                data.append(dict)

            except ValueError:
                print('value error')
    output_file = open(resource_filename(__name__, config['pos_feature_file_path']['path']), 'w', encoding='utf-8')
    for dic in data:
        json.dump(dic, output_file)
        output_file.write("\n")


if __name__ == "__main__":
    main()
