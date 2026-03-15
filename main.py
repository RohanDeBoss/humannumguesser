import os
import time
import pygame
import threading
import tkinter as tk
from tkinter import ttk
import numpy as np
from collections import deque
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from data import dataset, firstdataset, seconddataset, testsample, frequency, frequency2
import warnings
warnings.filterwarnings("ignore")

# ── File paths ────────────────────────────────────────────────────────────────
_dir = os.path.dirname(os.path.abspath(__file__))
assets                 = os.path.join(_dir, "assets") + os.sep
checknumberbutton      = assets + os.path.join("images", "check.png")
standardizedtestbutton = assets + os.path.join("images", "run907.png")
correctsfx             = assets + os.path.join("audios", "correct.mp3")
wrongsfx               = assets + os.path.join("audios", "wrong.mp3")

# ── Global state ──────────────────────────────────────────────────────────────
inputted  = []
firstinp  = []
secondinp = []
win       = 0
played    = []
confidence = {str(i).zfill(2): 0 for i in range(0, 101)}

# ── Persistent combined sequences ────────────────────────────────────────────
# Initialised from base datasets at startup, then extended by one entry per
# round. This eliminates the biggest per-round cost from doc2: rebuilding
# list(firstdataset) + list(firstinp) (~20k entries) every single call.
_first_combined  = list(firstdataset)
_second_combined = list(seconddataset)
_full_combined   = list(dataset)

# ── Dataset pattern matching index ───────────────────────────────────────────
_dataset_pos_index: dict = {}
for _i, _v in enumerate(dataset):
    _dataset_pos_index.setdefault(_v, []).append(_i)

# ── User history index ────────────────────────────────────────────────────────
_inp_index: dict = {}

# ── 907 test ──────────────────────────────────────────────────────────────────
_test_running = False

# ── ML config ─────────────────────────────────────────────────────────────────
# _ML_WINDOW = 1000 is the proven sweet spot from testing:
# 500=8.6, 1000=9.1, 2000=8.8, 3000=8.1, 4000=8.6
_ML_WINDOW = 1000
_N_LAGS    = 5
_RF_TREES  = 10
_XGB_WIN   = 10
_XGB_TREES = 15
_XGB_DEPTH = 10
_XGB_LR    = 0.11

# ── Model caches — retrained per round on windowed combined sequence ──────────
# The window slides over the combined (base + user) sequence, so as the test
# progresses the RF/XGB increasingly reflects the current person's patterns.
# This is what doc2 was doing and is the main driver of accuracy.
_rf_first_cache   = {"model": None, "train_len": -1}
_rf_second_cache  = {"model": None, "train_len": -1}
_rf_full_cache    = {"model": None, "train_len": -1}
_xgb_first_cache  = {"model": None, "train_len": -1}
_xgb_second_cache = {"model": None, "train_len": -1}

# ── Markov caches — order 1 only, incremental O(1) updates ───────────────────
# Orders 2 and 3 consistently hurt accuracy in testing — too sparse/noisy.
_markov = {k: {"chain": None, "train_len": -1} for k in [
    "first_o1", "second_o1", "full_o1",
]}


# ═════════════════════════════════════════════════════════════════════════════
# ML helpers
# ═════════════════════════════════════════════════════════════════════════════

def _prepare_rf(seq, n_lags):
    s = [float(x) for x in seq]
    X, y = [], []
    for i in range(len(s) - n_lags):
        X.append(s[i:i + n_lags])
        y.append(s[i + n_lags])
    return np.array(X), np.array(y)


def _predict_rf(sequence, cache, n_lags=_N_LAGS):
    """Retrain on sequence[-_ML_WINDOW:] if data changed, then predict."""
    seq = sequence[-_ML_WINDOW:]
    if len(seq) < n_lags + 1:
        raise ValueError("short")
    if cache["train_len"] != len(seq):
        X, y = _prepare_rf(seq, n_lags)
        if X.size == 0 or y.size == 0:
            raise ValueError("short")
        m = RandomForestRegressor(n_estimators=_RF_TREES, random_state=42, n_jobs=-1)
        m.fit(X, y)
        cache["model"]     = m
        cache["train_len"] = len(seq)
    last = np.array([float(x) for x in seq[-n_lags:]]).reshape(1, -1)
    return cache["model"].predict(last)[0]


def _predict_xgb(sequence, cache, window=_XGB_WIN):
    """Retrain on sequence[-_ML_WINDOW:] if data changed, then predict."""
    seq = sequence[-_ML_WINDOW:]
    if len(seq) <= window:
        raise ValueError("short")
    if cache["train_len"] != len(seq):
        s = [float(x) for x in seq]
        X_t, y_t = [], []
        for i in range(len(s) - window):
            g = np.array(s[i:i + window])
            X_t.append([np.mean(g), np.std(g), np.median(g),
                         np.max(g), np.min(g), np.max(g) - np.min(g)])
            y_t.append(s[i + window])
        m = xgb.XGBRegressor(n_estimators=_XGB_TREES, max_depth=_XGB_DEPTH,
                              learning_rate=_XGB_LR, objective='reg:squarederror')
        m.fit(np.array(X_t), np.array(y_t))
        cache["model"]     = m
        cache["train_len"] = len(seq)
    g = np.array([float(x) for x in seq[-window:]])
    feat = np.array([[np.mean(g), np.std(g), np.median(g),
                      np.max(g), np.min(g), np.max(g) - np.min(g)]])
    return int(cache["model"].predict(feat)[0])


# ═════════════════════════════════════════════════════════════════════════════
# Markov helpers
# ═════════════════════════════════════════════════════════════════════════════

def _build_markov(data, order):
    chain = {}
    for i in range(len(data) - order):
        state = tuple(data[i:i + order])
        nxt   = data[i + order]
        chain.setdefault(state, {}).setdefault(nxt, 0)
        chain[state][nxt] += 1
    return chain


def _update_markov(chain, data, order):
    if len(data) < order + 1:
        return chain
    state = tuple(data[-(order + 1):-1])
    nxt   = data[-1]
    chain.setdefault(state, {}).setdefault(nxt, 0)
    chain[state][nxt] += 1
    return chain


def _markov_pred(chain, data, order):
    state = tuple(data[-order:])
    while state not in chain and len(state) > 1:
        state = state[1:]
    if state in chain:
        trans = chain[state]
        total = sum(trans.values())
        if total > 0:
            return max(trans, key=lambda s: trans[s] / total)
    overall = {}
    for trans in chain.values():
        for s, c in trans.items():
            overall[s] = overall.get(s, 0) + c
    return max(overall, key=overall.get) if overall else None


def _sync_markov(key, data, order):
    c = _markov[key]
    if c["train_len"] == -1:
        c["chain"]     = _build_markov(data, order)
        c["train_len"] = len(data)
    elif c["train_len"] != len(data):
        c["chain"]     = _update_markov(c["chain"], data, order)
        c["train_len"] = len(data)


# ═════════════════════════════════════════════════════════════════════════════
# Confidence scoring
# ═════════════════════════════════════════════════════════════════════════════

def normal_pdf(x, mean, sigma):
    return (1.0 / (sigma * (2 * 3.141592653589793) ** 0.5)) * \
           (2.718281828459045 ** (-((x - mean) ** 2) / (2 * sigma ** 2)))


def normaldist(fd, sd, weight):
    global confidence
    for key in confidence:
        k_fd = 10 if key == "100" else int(key[0])
        k_sd = int(key[1])
        confidence[key] += normal_pdf(abs(k_fd - fd), 0, 2) * weight
        confidence[key] += (2 - abs(k_sd - sd) / 10) * weight
        if int(key) == fd * 10 + sd:
            confidence[key] += weight
        if k_sd == sd:
            confidence[key] += 0.5 * weight


def othernormaldist(target, weight):
    global confidence
    for key in confidence:
        confidence[key] += 10 * normal_pdf(abs(int(key) - target), 0, 10) * weight
    confidence[str(target).zfill(2)] += 0.35 * weight


# ═════════════════════════════════════════════════════════════════════════════
# differencepred — identical logic to doc2, but using persistent combined lists
# ═════════════════════════════════════════════════════════════════════════════

def differencepred():
    global confidence, firstinp, secondinp, inputted
    global _first_combined, _second_combined, _full_combined

    confidence = {str(i).zfill(2): 0 for i in range(0, 101)}
    if not inputted:
        return confidence

    # Extend persistent combined lists — O(1) instead of O(20k)
    try:
        fd_val = 10 if inputted[-1] == "100" else int(inputted[-1][0])
        sd_val = int(inputted[-1][1])
        firstinp.append(fd_val)
        secondinp.append(sd_val)
        _first_combined.append(fd_val)
        _second_combined.append(sd_val)
        _full_combined.append(inputted[-1])
    except:
        pass

    # ── RF on digit sequences (retrain on windowed combined) ──────────────────
    fd, sd = None, None
    try:
        fd = round(float(_predict_rf(_first_combined, _rf_first_cache)))
    except: pass
    if fd == 10:
        sd = 0
    elif fd is not None:
        try:
            sd = round(float(_predict_rf(_second_combined, _rf_second_cache)))
        except: pass
    if fd is not None and sd is not None:
        normaldist(fd, sd, 1.0)

    # ── Frequency table ────────────────────────────────────────────────────────
    try:
        fd2 = frequency[inputted[-1]][0]
        sd2 = 0 if fd2 == 10 else frequency[inputted[-1]][1]
        normaldist(fd2, sd2, 1.1)
    except: pass

    # ── Order-1 Markov on digit sequences ─────────────────────────────────────
    _sync_markov("first_o1",  _first_combined,  1)
    _sync_markov("second_o1", _second_combined, 1)
    try:
        fd = int(_markov_pred(_markov["first_o1"]["chain"],  _first_combined,  1))
        sd = 0 if fd == 10 else int(_markov_pred(_markov["second_o1"]["chain"], _second_combined, 1))
        normaldist(fd, sd, 1.7)
    except: pass

    # ── XGB on digit sequences (retrain on windowed combined) ─────────────────
    fd, sd = None, None
    try:
        fd = _predict_xgb(_first_combined, _xgb_first_cache)
    except: pass
    if fd == 100:
        sd = 0
    elif fd is not None:
        try:
            sd = _predict_xgb(_second_combined, _xgb_second_cache)
        except: pass
    if fd is not None and sd is not None:
        normaldist(fd, sd, 1.1)

    # ── RF on full sequence (retrain on windowed combined) ────────────────────
    try:
        pred = round(float(_predict_rf(_full_combined, _rf_full_cache)))
        othernormaldist(int(pred), 8.0)
    except: pass

    # ── Order-1 Markov on full sequence ───────────────────────────────────────
    _sync_markov("full_o1", _full_combined, 1)
    try:
        pred = int(_markov_pred(_markov["full_o1"]["chain"], _full_combined, 1))
        othernormaldist(pred, 4.6)
    except: pass

    # ── Frequency2 ────────────────────────────────────────────────────────────
    try:
        othernormaldist(int(frequency2[inputted[-1]]), 4.8)
    except: pass

    return confidence


# ═════════════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    global inputted, played, confidence, _inp_index

    confidence = differencepred()

    # ── Dataset pattern matching (index-accelerated) ──────────────────────────
    base_inc = (20609 + len(inputted)) / 7500000
    for val, positions in _dataset_pos_index.items():
        confidence[val] += base_inc * len(positions)

    if inputted:
        for i in _dataset_pos_index.get(inputted[-1], []):
            if i + 1 >= len(dataset):
                continue
            match_len = 1
            for depth in range(1, min(1001, i + 1, len(inputted))):
                if dataset[i - depth] == inputted[-1 - depth]:
                    match_len += 1
                else:
                    break
            if match_len >= 2:
                confidence[dataset[i + 1]] += 4.6 * match_len * (match_len - 1) / 2

    # ── User history pattern matching (index-accelerated) ────────────────────
    if len(inputted) >= 2:
        _inp_index.setdefault(inputted[-1], []).append(len(inputted) - 1)
        for i in _inp_index.get(inputted[-1], []):
            if i + 1 >= len(inputted):
                continue
            retro     = i / len(inputted)
            confidence[inputted[i]] += 0.7 * retro
            match_len = 1
            for depth in range(1, min(1001, i + 1, len(inputted))):
                if inputted[i - depth] == inputted[-1 - depth]:
                    match_len += 1
                else:
                    break
            if match_len >= 2:
                confidence[inputted[i + 1]] += 10.9 * retro * match_len * (match_len - 1) / 2

    # ── Arithmetic sequence detection ─────────────────────────────────────────
    if len(inputted) >= 2:
        diff = int(inputted[-1]) - int(inputted[-2])
        if diff in {1, 2, 3, 5, 10, 20, -1, -2, -3, -5, -10, -20}:
            ne = int(inputted[-1]) + diff
            if 0 <= ne <= 100:
                confidence[str(ne).zfill(2)] += 10

    if len(inputted) >= 3 and inputted[-1] != inputted[-2]:
        d1 = int(inputted[-1]) - int(inputted[-2])
        d2 = int(inputted[-2]) - int(inputted[-3])
        if d1 == d2:
            ne = int(inputted[-1]) + d1
            if 0 <= ne <= 100:
                confidence[str(ne).zfill(2)] += 30

    # ── Geometric sequence detection ──────────────────────────────────────────
    try:
        if len(inputted) >= 2:
            ratio = int(inputted[-2]) / int(inputted[-1])
            if ratio in {2, 0.5}:
                ne = int(int(inputted[-1]) * (int(inputted[-1]) / int(inputted[-2])))
                if 0 <= ne <= 100:
                    confidence[str(ne).zfill(2)] += 7
    except: pass

    try:
        if len(inputted) >= 3:
            ratios = [int(inputted[i]) / int(inputted[i - 1])
                      for i in range(len(inputted) - 3, len(inputted))]
            if all(r == ratios[0] for r in ratios):
                ne = int(int(inputted[-1]) * ratios[0])
                if 0 <= ne <= 100:
                    confidence[str(ne).zfill(2)] += 30
    except: pass

    if not inputted:
        return "37"

    try:
        if inputted[-1] == played[1] and inputted[-2] == played[2]:
            return played[0]
    except: pass

    if max(confidence.values()) == 0.0:
        return inputted[-1]

    return max(confidence, key=confidence.get)


# ═════════════════════════════════════════════════════════════════════════════
# UI helpers
# ═════════════════════════════════════════════════════════════════════════════

def start_ai_thread(input_text):
    def run():
        returned = main()
        root.after(0, update_ui_after_ai, input_text, returned)
    threading.Thread(target=run, daemon=True).start()


def update_ui_after_ai(input_text, returned):
    global win, played, inputted
    input_text = str(input_text).zfill(2) if 0 <= int(input_text) <= 9 else input_text
    inputted.append(input_text)
    played.insert(0, returned)
    if len(played) >= 4:
        played.pop(-1)
    if inputted[-1] == returned:
        pygame.mixer.music.load(correctsfx)
        pygame.mixer.music.play()
        result_label.config(text=f"    {returned}    ", bg="lawn green")
        win += 1
        winorloselabel.config(text="Bot Wins")
    else:
        pygame.mixer.music.load(wrongsfx)
        pygame.mixer.music.play()
        result_label.config(text=f"    {returned}    ", bg="red2")
        winorloselabel.config(text="Bot Lost")
    botplayedlabel.config(
        text=f"AI Win Rate: {(win / len(inputted) * 100):.3f}%\nRounds Played: {len(inputted)}")
    confidence_str = ""
    for key, value in confidence.items():
        confidence_str += f"{key}: {value:.2f}, "
        if int(key) % 6 == 0:
            confidence_str += "\n"
    confidencelabel.config(
        text=f"Confidence levels for prior number:\n{confidence_str}\n(don't use these to cheat weirdo)",
        fg='black', bg="pale turquoise")
    result_label.after(200, lambda: result_label.config(bg="skyblue1"))
    entry.focus_set()


def numinput(event=None):
    try:
        if len(inputted) >= 500:
            raise ValueError
        input_text = entry.get().strip()
        entry.delete(0, "end")
        result_label.config(text="            ")
        if not input_text.isdigit():
            raise ValueError
        val = int(input_text)
        if not (0 <= val <= 100):
            raise ValueError
        if len(input_text) > 1 and input_text[0] == "0":
            raise ValueError
        start_ai_thread(input_text)
    except ValueError:
        result_label.config(text="poopy number", bg="skyblue1")
    entry.focus_set()


# ═════════════════════════════════════════════════════════════════════════════
# 907 test
# ═════════════════════════════════════════════════════════════════════════════

def autonuminput(event=None):
    global _test_running
    if _test_running:
        return
    _test_running = True
    button907.config(state="disabled")

    test_state = {
        "win": 0, "streak": 0, "best_streak": 0,
        "lose_streak": 0, "worst_streak": 0,
        "history": [], "start_time": time.time(),
        "last_actual": "—", "last_guess": "—", "last_correct": None,
    }

    def update_live_ui(state, idx):
        total   = len(testsample)
        rate    = state["win"] / idx * 100 if idx > 0 else 0
        elapsed = time.time() - state["start_time"]
        rps     = idx / elapsed if elapsed > 0 else 0
        eta     = (total - idx) / rps if rps > 0 else 0
        progress_bar["value"] = idx
        progress_label.config(
            text=f"Round {idx}/{total}  —  {rate:.2f}% accuracy  —  {rps:.1f} rounds/sec  —  ETA {eta:.0f}s")
        streak_label.config(
            text=f"Current streak: {state['streak']}  |  Best win: {state['best_streak']}  |  Worst loss: {state['worst_streak']}")
        last_label.config(
            text=f"Actual: {state['last_actual']}   AI guessed: {state['last_guess']}",
            bg="lawn green" if state["last_correct"] else "red2")
        history_canvas.delete("all")
        for i, correct in enumerate(state["history"][-50:]):
            x0 = i * 14
            history_canvas.create_rectangle(
                x0, 0, x0 + 13, 20,
                fill="#22cc44" if correct else "#dd2222", outline="")
        botplayedlabel.config(text=f"AI Win Rate: {rate:.3f}%\nRounds Played: {idx}")

    def run_test():
        global win, inputted, _test_running
        for idx, input_text in enumerate(testsample, start=1):
            returned = main()
            inputted.append(input_text)
            correct = input_text == returned
            if correct:
                win += 1
                test_state["win"]    += 1
                test_state["streak"] += 1
                test_state["lose_streak"] = 0
                test_state["best_streak"] = max(test_state["best_streak"], test_state["streak"])
            else:
                test_state["lose_streak"] += 1
                test_state["streak"] = 0
                test_state["worst_streak"] = max(test_state["worst_streak"], test_state["lose_streak"])
            test_state["history"].append(correct)
            test_state["last_actual"]  = input_text
            test_state["last_guess"]   = returned
            test_state["last_correct"] = correct
            root.after(0, update_live_ui, dict(test_state), idx)

        def finish():
            global _test_running
            elapsed = time.time() - test_state["start_time"]
            rate    = test_state["win"] / len(testsample) * 100
            progress_label.config(
                text=f"Done! {len(testsample)} rounds in {elapsed:.1f}s  —  Final accuracy: {rate:.3f}%")
            last_label.config(text="Test complete", bg="pale turquoise")
            button907.config(state="normal")
            _test_running = False
        root.after(0, finish)

    threading.Thread(target=run_test, daemon=True).start()


# ═════════════════════════════════════════════════════════════════════════════
# UI init
# ═════════════════════════════════════════════════════════════════════════════

pygame.mixer.init()

root = tk.Tk()
root.title("Number Predictor Thing")
root.configure(bg="pale turquoise")
root.geometry("1280x620")
root.attributes("-fullscreen", True)

def toggle_fullscreen(event=None):
    root.attributes("-fullscreen", False)
root.bind("<Escape>", toggle_fullscreen)

top_frame    = tk.Frame(root, bg="pale turquoise")
top_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=10)
middle_frame = tk.Frame(root, bg="pale turquoise")
middle_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
bottom_frame = tk.Frame(root, bg="pale turquoise")
bottom_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=10)
right_frame  = tk.Frame(root, bg="pale turquoise")
right_frame.grid(row=0, column=1, rowspan=3, sticky="nsew", padx=20, pady=10)

root.grid_columnconfigure(0, weight=3)
root.grid_columnconfigure(1, weight=2)
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=2)
root.grid_rowconfigure(2, weight=1)

maintitle = tk.Label(top_frame, text="Number Predictor Thing",
                     font=("Helvetica", 40, "bold"), bg="white")
maintitle.pack(pady=10)
dataset_label = tk.Label(top_frame,
    text=f"dataset: {len(dataset)} | first: {len(firstdataset)} | "
         f"second: {len(seconddataset)} | test: {len(testsample)}  |  ML window: {_ML_WINDOW}",
    font=("Helvetica", 11), bg="pale turquoise")
dataset_label.pack()
entry = tk.Entry(top_frame, font=("Helvetica", 30))
entry.pack(pady=10)
entry.bind("<Return>", numinput)

img_button    = tk.PhotoImage(file=checknumberbutton)
img_907button = tk.PhotoImage(file=standardizedtestbutton)
check_button  = tk.Button(middle_frame, image=img_button,    borderwidth=0,
                           compound=tk.CENTER, bg="pale turquoise")
check_button.grid(row=0, column=0, padx=20, pady=20)
button907     = tk.Button(middle_frame, image=img_907button, borderwidth=0,
                           compound=tk.CENTER, bg="pale turquoise")
button907.grid(row=0, column=1, padx=20, pady=20)

stats_frame = tk.Frame(middle_frame, bg="pale turquoise")
stats_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=5)

progress_bar = ttk.Progressbar(stats_frame, orient="horizontal", length=600,
                                mode="determinate", maximum=len(testsample))
progress_bar.pack(pady=4)

progress_label = tk.Label(stats_frame,
    text="Press the 907 button to start the standardised test",
    font=("Helvetica", 13), bg="pale turquoise")
progress_label.pack()

streak_label = tk.Label(stats_frame,
    text="Current streak: 0  |  Best win: 0  |  Worst loss: 0",
    font=("Helvetica", 13, "bold"), bg="pale turquoise", fg="black")
streak_label.pack(pady=2)

last_label = tk.Label(stats_frame, text="Actual: —   AI guessed: —",
    font=("Helvetica", 14, "bold"), bg="pale turquoise", width=40)
last_label.pack(pady=2)

history_canvas = tk.Canvas(stats_frame, width=700, height=20,
                            bg="pale turquoise", highlightthickness=0)
history_canvas.pack(pady=4)

result_label   = tk.Label(bottom_frame, text="            ",
                            font=("Helvetica", 50), bg="skyblue1")
result_label.pack(pady=10)
winorloselabel = tk.Label(bottom_frame, text="",
                           font=("Helvetica", 50), bg="pale turquoise")
winorloselabel.pack(pady=10)
botplayedlabel = tk.Label(bottom_frame, text="AI Win Rate: NA%\nRounds Played: 0",
                           font=('Helvetica', 30, 'bold'), fg='black', bg="pale turquoise")
botplayedlabel.pack(pady=10)

confidenceinit = {str(i).zfill(2): 0 for i in range(0, 101)}
confidence_str = ""
for key, value in confidenceinit.items():
    confidence_str += f"{key}: {value:.2f}, "
    if int(key) % 6 == 0:
        confidence_str += "\n"
confidencelabel = tk.Label(right_frame,
    text=f"Confidence levels for prior number:\n{confidence_str}\n(don't use these to cheat weirdo)",
    fg='black', bg="pale turquoise", font=('Helvetica', 15, 'bold'))
confidencelabel.pack(pady=20)

check_button.bind("<Button-1>", numinput)
button907.bind("<Button-1>", autonuminput)

if __name__ == "__main__":
    root.mainloop()