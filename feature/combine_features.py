import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    ls_y = []
    for file in sorted(glob.glob('/Users/wenxu/PycharmProjects/BertModel/data/svm_result/2.*.csv'),
                       key=os.path.getmtime):
        df = pd.read_csv(file)
        y = df.iloc[4]['f1-score']
        print(y, file.split()[2])
        if y >= 0.59:
            ls_y.append(int(file.split()[3]))
    print(ls_y)
