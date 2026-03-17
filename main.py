# Version 2.3 - Final: if input_len >= 2: (count ** 2.5) * 1.5 and input_len >= 3: (count ** 3.5) * 2

# if input_len >= 3:
# 2.5 = 11.466
# 3.5 = 11.466
# 4.5 = 11.466
# 3 * 2.5 = 11.466

# if input_len >= 2:
# 2 = 11.466
# 2.5 = 11.577
# 3 = 11.577
# 2.5 * 2 = 11.466

#Try 5.2 = 11.1, 4.7 = 11.577 (KEEP) 4.6 = 11.577, 4.5 = 11.356, 4.4 = 11.356

import os
import pygame
import tkinter as tk
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from data import dataset, firstdataset, seconddataset, testsample, frequency, frequency2
import warnings
import time
import threading
from tkinter import ttk
from collections import defaultdict

warnings.filterwarnings("ignore")

_dir = os.path.dirname(os.path.abspath(__file__))
assets = os.path.join(_dir, "assets") + os.sep
checknumberbutton      = assets + os.path.join("images", "check.png")
standardizedtestbutton = assets + os.path.join("images", "run907.png")
correctsfx             = assets + os.path.join("audios", "correct.mp3")
wrongsfx               = assets + os.path.join("audios", "wrong.mp3")

global temp, tempc, next_element, confidence, nextfirstdiff, nextseconddiff
inputted, firstdiff, seconddiff, temp, tempc, win, train, firstinp, secondinp, played = [], [], [], [], [], 0, [], [], [], []

# Precompute dataset indices and counts
dataset_counts = defaultdict(int)
dataset_indices = defaultdict(list)
for i, val in enumerate(dataset):
    dataset_counts[val] += 1
    dataset_indices[val].append(i)

# ── Precompute base Markov chains once at startup ─────────────────────────────
# build_markov_chain on 20k entries was the main per-round bottleneck.
# These base chains are immutable; each round we copy and add only the tiny
# user-data tail (~round_number entries), which is O(user_len) not O(20k).
def _build_base_mc(data, k):
    mc = {}
    for i in range(len(data) - k):
        state = tuple(data[i:i+k])
        nxt   = data[i + k]
        if state not in mc: mc[state] = {}
        mc[state][nxt] = mc[state].get(nxt, 0) + 1
    return mc

_base_mc_first  = _build_base_mc(firstdataset,  1)
_base_mc_second = _build_base_mc(seconddataset, 1)
_base_mc_full   = _build_base_mc(dataset,       1)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data(sequence, n_lags=2):
    arr = np.array(sequence)
    X = np.column_stack([arr[i : len(arr) - n_lags + i] for i in range(n_lags)])
    y = arr[n_lags:]
    return X, y

def predict_next(sequence, n_lags=2):
    if len(sequence) < n_lags + 1: raise ValueError("short")
    X, y = prepare_data(sequence, n_lags)
    if X.size == 0 or y.size == 0: raise ValueError("short")
    model = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=None)
    model.fit(X, y)
    last_values = np.array(sequence[-n_lags:]).reshape(1, -1)
    return model.predict(last_values)[0]

def get_xgb_features(inp_array, window_size=10):
    arr = np.array(inp_array, dtype=float)
    N = len(arr)
    if N <= window_size:
        return np.empty((0, 6)), np.empty((0,)), np.empty((0, 6))
    windows = np.array([arr[i:i+window_size] for i in range(N - window_size)])
    y_train = arr[window_size:]
    means   = np.mean(windows, axis=1)
    stds    = np.std(windows,  axis=1)
    medians = np.median(windows, axis=1)
    maxs    = np.max(windows,  axis=1)
    mins    = np.min(windows,  axis=1)
    ranges  = maxs - mins
    X_train = np.column_stack((means, stds, medians, maxs, mins, ranges))
    ng = arr[-window_size:]
    X_pred = np.array([[np.mean(ng), np.std(ng), np.median(ng), np.max(ng), np.min(ng), np.max(ng)-np.min(ng)]])
    return X_train, y_train, X_pred

def normal_pdf(x, mean, sigma):
    factor   = 1 / (sigma * (2 * 3.141592653589793)**0.5)
    exponent = -((x - mean)**2) / (2 * sigma**2)
    return factor * (2.718281828459045**exponent)

def normaldist(target_first_digit, target_second_digit, weight):
    global confidence
    for key in confidence.keys():
        if key != "100": first_digit = int(key[0])
        else: first_digit = 10
        distance = abs(first_digit - target_first_digit)
        confidence[key] += (normal_pdf(distance, 0, 2)) * weight
        second_digit = int(key[1])
        distance = abs(second_digit - target_second_digit)
        confidence[key] += (2 - (distance / 10)) * weight
        if int(key) == (target_first_digit * 10 + target_second_digit):
            confidence[key] += 1 * weight
        if int(key[1]) == (target_second_digit):
            confidence[key] += 0.5 * weight
    return confidence

def othernormaldist(target_number, weight):
    global confidence
    for key in confidence.keys():
        number = int(key)
        distance = abs(number - target_number)
        confidence[key] += (10 * normal_pdf(distance, 0, 10)) * weight
    for key in confidence.keys():
        number = int(key)
        if number == target_number:
            confidence[key] += 0.35 * weight

def build_markov_chain(data, k):
    markov_chain = defaultdict(lambda: defaultdict(int))
    for i in range(len(data) - k):
        current_state = tuple(data[i:i+k])
        next_state = data[i + k]
        markov_chain[current_state][next_state] += 1
    return markov_chain

def predict_next_elementmark(markov_chain, current_state):
    while current_state not in markov_chain and len(current_state) > 1:
        current_state = current_state[1:]
    if current_state in markov_chain:
        transitions = markov_chain[current_state]
        if transitions:
            return max(transitions, key=transitions.get)
    overall_transitions = defaultdict(int)
    for transitions in markov_chain.values():
        for next_state, count in transitions.items():
            overall_transitions[next_state] += count
    if overall_transitions:
        return max(overall_transitions, key=overall_transitions.get)
    return None

def _mc_from_base(base_mc, user_data, k):
    """Copy base chain (plain dicts, fast) then add user_data transitions."""
    mc = {state: dict(nexts) for state, nexts in base_mc.items()}
    seq = user_data  # user_data is the full user list (small)
    # add transitions that bridge the base tail into user data
    # we need the last k items of the base dataset plus all user entries
    for i in range(len(seq) - k):
        state = tuple(seq[i:i+k])
        nxt   = seq[i + k]
        if state not in mc: mc[state] = {}
        mc[state][nxt] = mc[state].get(nxt, 0) + 1
    return mc

def differencepred():
    global nextfirstdiff, nextseconddiff, confidence, firstinp, secondinp, inputted
    confidence = {str(i).zfill(2): 0 for i in range(0, 101)}
    if len(inputted) == 0: return confidence
    try:
        if inputted[-1] == "100": firstinp.append(10)
        else: firstinp.append(int(inputted[-1][0]))
        secondinp.append(int(inputted[-1][1]))
    except: pass

    first_train  = firstdataset  + firstinp
    second_train = seconddataset + secondinp
    main_train   = dataset       + inputted

    nextfirstdiff, nextseconddiff = None, None
    try: nextfirstdiff = round(float(predict_next(first_train)))
    except ValueError: pass
    if nextfirstdiff == 10: nextseconddiff = 0
    else:
        try: nextseconddiff = round(float(predict_next(second_train)))
        except ValueError: pass
    if nextseconddiff and nextfirstdiff: normaldist(nextfirstdiff, nextseconddiff, 1)
    nextfirstdiff, nextseconddiff = None, None
    try:
        nextfirstdiff = frequency[inputted[-1]][0]
        if nextfirstdiff == 10: nextseconddiff = 0
        else: nextseconddiff = frequency[inputted[-1]][1]
        normaldist(nextfirstdiff, nextseconddiff, 1.1)
    except: pass
    nextfirstdiff, nextseconddiff = None, None

    # Markov first/second — copy precomputed base + add user tail only
    try:
        mc = _mc_from_base(_base_mc_first, firstinp, 1)
        current_state = tuple(first_train[-1:])
        nextfirstdiff = int(predict_next_elementmark(mc, current_state))
    except: pass
    if nextfirstdiff == 10: nextseconddiff = 0
    else:
        try:
            mc = _mc_from_base(_base_mc_second, secondinp, 1)
            current_state = tuple(second_train[-1:])
            nextseconddiff = int(predict_next_elementmark(mc, current_state))
        except: pass
    if nextseconddiff and nextfirstdiff: normaldist(nextfirstdiff, nextseconddiff, 1.7)
    nextfirstdiff, nextseconddiff = None, None

    try:
        X_train, y_train, X_pred = get_xgb_features(firstinp, 10)
        if len(y_train) > 0:
            model = xgb.XGBRegressor(n_estimators=35, max_depth=10, learning_rate=0.11, objective='reg:squarederror', n_jobs=1)
            model.fit(X_train, y_train)
            nextfirstdiff = int(model.predict(X_pred)[0])
    except: pass
    if nextfirstdiff == 100: nextseconddiff = 0
    else:
        try:
            X_train, y_train, X_pred = get_xgb_features(secondinp, 10)
            if len(y_train) > 0:
                model = xgb.XGBRegressor(n_estimators=35, max_depth=10, learning_rate=0.11, objective='reg:squarederror', n_jobs=1)
                model.fit(X_train, y_train)
                nextseconddiff = int(model.predict(X_pred)[0])
        except: pass
    if nextseconddiff and nextfirstdiff: normaldist(nextfirstdiff, nextseconddiff, 1.1)

    nextfirstdiff = None
    try: nextfirstdiff = round(float(predict_next(main_train)))
    except: pass
    if nextseconddiff: othernormaldist(int(nextfirstdiff), 8)
    nextfirstdiff = None

    # Full Markov — copy precomputed base + add user tail only
    try:
        mc = _mc_from_base(_base_mc_full, inputted, 1)
        current_state = tuple(main_train[-1:])
        nextfirstdiff = int(predict_next_elementmark(mc, current_state))
    except: pass
    if nextfirstdiff: othernormaldist(int(nextfirstdiff), 4.6)
    nextfirstdiff = None
    try: nextfirstdiff = frequency2[inputted[-1]]
    except: pass
    if nextfirstdiff: othernormaldist(int(nextfirstdiff), 4.7) #Try 5.2 = 11.1, 4.7 = 11.577 (KEEP) 4.6 = 11.577, 4.5 = 11.356, 4.4 = 11.356
    return confidence

def main():
    global inputted, retro, temp, tempc, next_element, confidence, firstinp, secondinp
    next_element, difference = 0, 0
    confidence = differencepred()

    input_len = len(inputted)
    base_add  = (20609 + input_len) / 7500000

    for val, count in dataset_counts.items():
        if val in confidence:
            confidence[val] += count * base_add

    if input_len > 0:
        last_val = inputted[-1]
        for i in dataset_indices.get(last_val, []):
            j_limit = len(dataset) - i
            if j_limit > 1000002: j_limit = 1000002
            if j_limit > 2:
                L = 1
                while L < j_limit - 1 and L < input_len:
                    if dataset[i - L] == inputted[-1 - L]: L += 1
                    else: break
                if L >= 2:
                    confidence[dataset[i + 1]] += (L * (L - 1) / 2) * 4.6

    if input_len > 0:
        last_val = inputted[-1]
        for i in range(input_len):
            retro = i / input_len #Decay makes it worse
            confidence[inputted[i]] += 0.7 * retro
            if inputted[i] == last_val:
                j_limit = input_len - i
                if j_limit > 1000002: j_limit = 1000002
                if j_limit > 2:
                    L = 1
                    while L < j_limit - 1 and L < input_len:
                        if inputted[i - L] == inputted[-1 - L]: L += 1
                        else: break
                    if L >= 2:
                        confidence[inputted[i + 1]] += (L * (L - 1) / 2) * 11 * retro #11 = 11.577, 12 = 11.577

    if (len(inputted) >= 2) and (int(inputted[-2]) - int(inputted[-1]) in {1, 2, 3, 5, 10, 20, -1, -2, -3, -5, -10, -20}):
        next_element = int(inputted[-1]) + (int(inputted[-1]) - int(inputted[-2]))
        if (0 <= next_element <= 9): next_element = f"0{next_element}"
        if (0 <= int(next_element) <= 100): confidence[str(next_element)] += 10
    if (len(inputted) >= 3) and (inputted[-1] != inputted[-2]) and (int(inputted[-1]) - int(inputted[-2])) == (int(inputted[-2]) - int(inputted[-3])):
        difference = int(inputted[-1]) - int(inputted[-2])
        next_element = int(inputted[-1]) + difference
        if (0 <= next_element <= 9): next_element = f"0{next_element}"
        if (0 <= int(next_element) <= 100): confidence[str(next_element)] += 30
    try:
        if (len(inputted) >= 2) and ((int(inputted[-2])/int(inputted[-1])) in {2, 0.5}):
            next_element = int(int(inputted[-1]) * (int(inputted[-1]) / int(inputted[-2])))
            if (0 <= int(next_element) <= 9): next_element = f"0{next_element}"
            if (0 <= int(next_element) <= 100): confidence[str(next_element)] += 7
    except: pass
    try:
        ratios = [int(inputted[i]) / int(inputted[i-1]) for i in range(len(inputted)-3, len(inputted))]
        if all(ratio == ratios[0] for ratio in ratios):
            next_element = int((int(inputted[-1])) * ratios[0])
            if (0 <= next_element <= 9): next_element = f"0{next_element}"
            if (0 <= int(next_element) <= 100): confidence[str(next_element)] += 30
    except: pass
    try:
        if len(inputted) >= 5 and (int(inputted[-1]) - int(inputted[-3])) == (int(inputted[-3]) - int(inputted[-5])):
            next_element = int(inputted[-2]) + (int(inputted[-2]) - int(inputted[-4]))
            if (0 <= next_element <= 9): next_element = f"0{next_element}"
            if (0 <= int(next_element) <= 100): confidence[str(next_element)] += 30
    except: pass

    try:
        if input_len >= 2:
            last_1 = inputted[-1]
            freq_1 = defaultdict(int)
            for i in range(input_len - 1):
                if inputted[i] == last_1:
                    freq_1[inputted[i+1]] += 1
            for next_val, count in freq_1.items():
                if count >= 2:
                    confidence[next_val] += (count ** 2.5) * 1.5
        if input_len >= 3:
            last_val_1 = inputted[-1]
            last_val_2 = inputted[-2]
            freq_2 = defaultdict(int)
            for i in range(input_len - 2):
                if inputted[i] == last_val_2 and inputted[i+1] == last_val_1:
                    freq_2[inputted[i+2]] += 1
            for next_val, count in freq_2.items():
                if count >= 2:
                    confidence[next_val] += (count ** 3.5) * 2
    except: pass

    if (len(inputted)) == 0: return "37"
    try:
        if (inputted[-1] == played[1]) and (inputted[-2] == played[2]): return played[0]
    except: pass
    if max(confidence.items()) == 0.0: return inputted[-1]
    inverted_confidence = {v: k for k, v in confidence.items()}
    return inverted_confidence[max(confidence.values())]


def numinput(event=None):
    global win, confidence, played, timerup, inputted
    try:
        if (timerup == False) and (len(inputted) < 500): input_text = entry.get()
        else: raise ValueError
        entry.delete(0, "end")
        result_label.config(text="            ")
        if (0 <= int(input_text) <= 100) and ((((input_text[0] not in {"0", " "}) == (0 <= int(input_text))<= 100)) or input_text == "0"):
            returned = main()
            inputted.append(input_text)
            if (0 <= int(inputted[-1]) <= 9): inputted[-1] = f"0{inputted[-1]}"
            played.insert(0, returned)
            if len(played) >= 4: played.pop(-1)
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
            botplayedlabel.config(text=f"AI Win Rate: {(win/len(inputted)*100):.3f}%\nRounds Played: {len(inputted)}")
            result_label.after(200, lambda: result_label.config(bg="skyblue1"))
            confidence_str = ""
            for key, value in confidence.items():
                confidence_str += f"{key}: {value:.2f}, "
                if int(key) % 6 == 0:
                    confidence_str += "\n"
            confidencelabel.config(text=f"Confidence levels for prior number:\n{confidence_str}\n(don't use these to cheat weirdo)", fg='black', bg="pale turquoise")
        else: raise ValueError
    except ValueError: result_label.config(text="poopy number", bg="skyblue1")
    entry.focus_set()


def autonuminput(event=None):
    global _test_running
    if _test_running: return
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
        progress_label.config(text=f"Round {idx}/{total}  —  {rate:.2f}% accuracy  —  {rps:.1f} rounds/sec  —  ETA {eta:.0f}s")
        streak_label.config(text=f"Current streak: {state['streak']}  |  Best win: {state['best_streak']}  |  Worst loss: {state['worst_streak']}")
        last_label.config(text=f"Actual: {state['last_actual']}   AI guessed: {state['last_guess']}", bg="lawn green" if state["last_correct"] else "red2")
        history_canvas.delete("all")
        for i, correct in enumerate(state["history"][-50:]):
            x0 = i * 14
            history_canvas.create_rectangle(x0, 0, x0+13, 20, fill="#22cc44" if correct else "#dd2222", outline="")
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
            progress_label.config(text=f"Done! {len(testsample)} rounds in {elapsed:.1f}s  —  Final accuracy: {rate:.3f}%")
            last_label.config(text="Test complete", bg="pale turquoise")
            button907.config(state="normal")
            _test_running = False
        root.after(0, finish)
    threading.Thread(target=run_test, daemon=True).start()


_test_running = False
pygame.mixer.init()
timerup = False
root = tk.Tk()
root.title("Number predictor thing")
root.configure(bg="pale turquoise")
root.attributes("-fullscreen", True)

def toggle_fullscreen(event=None):
    root.attributes("-fullscreen", False)
root.bind("<Escape>", toggle_fullscreen)

root.grid_columnconfigure(0, weight=3)
root.grid_columnconfigure(1, weight=2)
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=2)
root.grid_rowconfigure(2, weight=1)

top_frame    = tk.Frame(root, bg="pale turquoise")
top_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=10)
middle_frame = tk.Frame(root, bg="pale turquoise")
middle_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
bottom_frame = tk.Frame(root, bg="pale turquoise")
bottom_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=10)
right_frame  = tk.Frame(root, bg="pale turquoise")
right_frame.grid(row=0, column=1, rowspan=3, sticky="nsew", padx=20, pady=10)

maintitle = tk.Label(top_frame, text="Number Predictor Thing", font=("Helvetica", 40, "bold"), bg="white")
maintitle.pack(pady=10)
entry = tk.Entry(top_frame, font=("Helvetica", 30))
entry.pack(pady=10)
entry.bind("<Return>", numinput)
entry.focus_set()

img_button    = tk.PhotoImage(file=checknumberbutton)
img_907button = tk.PhotoImage(file=standardizedtestbutton)
check_button  = tk.Button(middle_frame, image=img_button,    borderwidth=0, compound=tk.CENTER, bg="pale turquoise")
check_button.grid(row=0, column=0, padx=20, pady=20)
button907     = tk.Button(middle_frame, image=img_907button, borderwidth=0, compound=tk.CENTER, bg="pale turquoise")
button907.grid(row=0, column=1, padx=20, pady=20)

stats_frame = tk.Frame(middle_frame, bg="pale turquoise")
stats_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
progress_bar = ttk.Progressbar(stats_frame, orient="horizontal", length=600, mode="determinate", maximum=len(testsample))
progress_bar.pack(pady=4)
progress_label = tk.Label(stats_frame, text="Press the 907 button to start the standardised test", font=("Helvetica", 13), bg="pale turquoise")
progress_label.pack()
streak_label = tk.Label(stats_frame, text="Current streak: 0  |  Best win: 0  |  Worst loss: 0", font=("Helvetica", 13, "bold"), bg="pale turquoise", fg="black")
streak_label.pack(pady=2)
last_label = tk.Label(stats_frame, text="Actual: —   AI guessed: —", font=("Helvetica", 14, "bold"), bg="pale turquoise", width=40)
last_label.pack(pady=2)
history_canvas = tk.Canvas(stats_frame, width=700, height=20, bg="pale turquoise", highlightthickness=0)
history_canvas.pack(pady=4)

result_label   = tk.Label(bottom_frame, text="            ", font=("Helvetica", 50), bg="skyblue1")
result_label.pack(pady=10)
winorloselabel = tk.Label(bottom_frame, text="", font=("Helvetica", 50), bg="pale turquoise")
winorloselabel.pack(pady=10)
botplayedlabel = tk.Label(bottom_frame, text="AI Win Rate: NA%\nRounds Played: 0", font=('Helvetica', 30, 'bold'), fg='black', bg="pale turquoise")
botplayedlabel.pack(pady=10)

confidenceinit = {str(i).zfill(2): 0 for i in range(0, 101)}
confidence_str = ""
for key, value in confidenceinit.items():
    confidence_str += f"{key}: {value:.2f}, "
    if int(key) % 6 == 0:
        confidence_str += "\n"
confidencelabel = tk.Label(right_frame, text=f"Confidence levels for prior number:\n{confidence_str}\n(don't use these to cheat weirdo)", fg='black', bg="pale turquoise", font=('Helvetica', 11, 'bold'), justify="left")
confidencelabel.pack(pady=20)

check_button.bind("<Button-1>", numinput)
button907.bind("<Button-1>", autonuminput)

if __name__ == "__main__":
    root.mainloop()