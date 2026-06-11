from pathlib import Path
import time
ns = {'__file__': str(Path('main.py').resolve()), '__name__': 'not_main'}
text = Path('main.py').read_text(encoding='utf-8')
exec(text.split('# UI BUILD')[0], ns)
from data import testsample
from my_dataset import my_testsample

def score(sample):
    ns['inputted'], ns['firstinp'], ns['secondinp'] = [], [], []
    wins = 0
    for actual in sample:
        guess = ns['main']()
        ns['inputted'].append(actual)
        wins += guess == actual
    return wins, len(sample)

t0 = time.time()
r907 = score(testsample)
t1 = time.time()
rmy = score(my_testsample)
t2 = time.time()

print(f'907 {r907[0]}/{r907[1]} ({100*r907[0]/r907[1]:.2f}%) - {t1-t0:.1f}s')
print(f'my  {rmy[0]}/{rmy[1]} ({100*rmy[0]/rmy[1]:.2f}%) - {t2-t1:.1f}s')
print(f'total {t2-t0:.1f}s')
