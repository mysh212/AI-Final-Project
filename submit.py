from lib.type import ds
from lib.model import get_model as gm
from lib.test import run_tests as rt
from lib.encode import encode
from lib.submit import submit
from lib.env import describe

def kaggle_submit(describe: str = ''):
    f = ds('data/train')
    encode(rt(gm(len(f.mark))), f.mark)
    submit(describe)
    return

if __name__ == '__main__':
    kaggle_submit(describe())
