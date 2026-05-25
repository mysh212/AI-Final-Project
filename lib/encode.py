from core.general import *

import pandas as pd

def encode(x: dict, mark):
    # debug(x)
    f = [[j, f'{j}.jpg', mark[i]] for j, i in sorted(list(zip(map(int, x.keys()), x.values())))]
    df = pd.DataFrame(f, columns = 'id filename label'.split())

    df.to_csv('data/ans.csv', index = False)
    return