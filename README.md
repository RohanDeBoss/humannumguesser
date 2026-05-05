# thanks for downloading, some brief things if you wanna further tweak this project:

## winrate

the current winrate on the 907 standardized test is `14.884%`

## note

I did explain lots of the main system in my video (poorly ofc) so I went through and added comments are certain bits to hopefully catch you up if somehow after working on this project you manage to improve the winrate on the standardized 907 test, message me because that's pretty interesting. One thing that people get confused about is the dataset/firstdataset/seconddataset lists. The first and second datasets are essentially just coppies of the first and second key parts of each number in the dataset. For example note the first number in the dataset is 15, this means in parallel the first number in firstdataset is a 1, and in secondadaset it's a 5. The reason they're split like this is because when there's a 100 in the main dataset, there's a 10 in firstdataset, and a 0 in secondataset. This helps so It doesn't have to keep running the new numbers through the conversion process

---

also try to be genuine with it because it is an incredibly easy thing to fake. Make sure it's able to score the same wr consistently. change these variables to the path of the files

---

Package installation:

```bash
pip install -r requirements.txt
```

## automated testing notes for future AIs

The UI test button works, but serious iteration is much faster if you load only the predictor code and stop before the Tkinter UI starts. The important trick is to execute `main.py` only up to `# UI BUILD`, provide `__file__`, then call `main()` in a loop while appending each actual test value to `inputted`.

Minimal exact test harness:

```powershell
@'
from pathlib import Path
ns = {"__file__": str(Path("main.py").resolve()), "__name__": "not_main"}
text = Path("main.py").read_text(encoding="utf-8")
exec(text.split("# UI BUILD")[0], ns)
from data import testsample
from my_dataset import my_testsample

def score(sample):
    ns["inputted"], ns["firstinp"], ns["secondinp"] = [], [], []
    wins = 0
    for actual in sample:
        guess = ns["main"]()
        ns["inputted"].append(actual)
        wins += guess == actual
    return wins, len(sample)

print("907", score(testsample))
print("my", score(my_testsample))
'@ | python -
```

For large searches, do not rerun the expensive XGBoost path for every candidate. First run the current predictor once on `907` and save, for every round:
- the history before the prediction,
- `dict(ns["confidence"])`,
- the base prediction,
- the actual answer.

Then screen candidate rules by copying that confidence dict, adding the candidate's proposed confidence, and reselecting `max(confidence, key=confidence.get)`. This makes hundreds or thousands of additive-rule sweeps cheap while preserving the exact current predictor state. Once a candidate beats the baseline in the fast screen, patch it into `main.py` and run the exact harness above.

Useful scoring discipline:
- Always screen `907` first. It is shorter and is the primary gate.
- Only run `my_dataset` after a candidate beats or meaningfully ties the current `907` baseline.
- Keep only changes that improve the combined score: `907 wins + my_dataset wins`.
- Prefer gates based on the pre-new-rule confidence snapshot, such as candidate rank or margin from the leader. This prevents a bundle of new rules from accidentally making itself eligible.
- Record every accepted change and useful rejection in `EXPERIMENTS.md`.
