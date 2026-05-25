from lib.type import ds
from lib.model import get_model as gm
from lib.test import run_tests as rt
from lib.encode import encode
from lib.submit import submit

def kaggle_submit():
    f = ds('data/train')
    encode(rt(gm(len(f.mark))), f.mark)
    submit()

if __name__ == '__main__':
    kaggle_submit()
