from core.general import *
from core.log import log as _log

import os

log = _log('submit')

def submit(description: str = ''):
    log.info('Submit')(f'Submitting', description = description)
    os.system(f'kaggle competitions submit -c 2026-nckuee-ai-course-x-ray-classification -f data/ans.csv  -m "{description}"')
    return

if __name__ == '__main__':
    submit()
    quit()