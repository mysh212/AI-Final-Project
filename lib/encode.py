from core.general import *

import pandas as pd

def encode(x, mark):
    f = [[j, f'{j}.jpg', mark[i]] for j, i in enumerate(x)]
    df = pd.DataFrame(f, columns = 'id filename label'.split())

    df.to_csv('data/ans.csv', index = False)
    return