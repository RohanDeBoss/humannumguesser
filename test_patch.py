from pathlib import Path
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

print('907', score(testsample))
print('my', score(my_testsample))
