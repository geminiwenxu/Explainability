import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    ls_y = []
    counter = 0
    for file in sorted(glob.glob('/Users/wenxu/PycharmProjects/BertModel/data/svm_result_2/4*.csv'), key=os.path.getmtime):
        print(file)
        df = pd.read_csv(file)
        y = df.iloc[4]['f1-score']
        ls_y.append(y)
        counter += 1
    x = np.arange(counter)
    fig = plt.figure()
    ax = plt.axes()
    plt.ylabel('f1-score')
    plt.xlabel('features')
    ax.plot(x, ls_y)
    plt.show()
