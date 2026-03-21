# Version 3.9 stronger retro sequence + diff/ratio rules
#907: 12.679 -> 12.789
#my: 7.85 -> 7.95

import os
import glob
import pygame
import tkinter as tk
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from data_filtered_v4 import dataset_filtered as dataset
from data_filtered_v4 import firstdataset_filtered as firstdataset
from data_filtered_v4 import seconddataset_filtered as seconddataset
from data import testsample, frequency, frequency2
import warnings
import time
import threading
from tkinter import ttk
from collections import defaultdict

warnings.filterwarnings("ignore")

_dir = os.path.dirname(os.path.abspath(__file__))
assets = os.path.join(_dir, "assets") + os.sep
checknumberbutton      = assets + os.path.join("images", "check.png")
standardizedtestbutton = assets + os.path.join("images", "runsettest.png")
correctsfx             = assets + os.path.join("audios", "correct.mp3")
wrongsfx               = assets + os.path.join("audios", "wrong.mp3")

global temp, tempc, next_element, confidence, nextfirstdiff, nextseconddiff
inputted, firstdiff, seconddiff, temp, tempc, win, train, firstinp, secondinp, played = [], [], [], [], [], 0, [], [], [], []

# ── Dataset management ────────────────────────────────────────────────────────
_create_mode     = False
_current_entries = []
BUILTIN_907      = "907 Standard"

def _dataset_path(name):
    return os.path.join(_dir, f"{name}_dataset.py")

def _save_dataset(name, entries):
    with open(_dataset_path(name), "w") as f:
        f.write(f"my_testsample = {entries!r}\n")

def _load_dataset(name):
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, _dataset_path(name))
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return list(mod.my_testsample)
    except:
        return []

def _scan_saved_datasets():
    paths = glob.glob(os.path.join(_dir, "*_dataset.py"))
    return [os.path.basename(p).replace("_dataset.py", "") for p in sorted(paths)]

def _reset_session():
    global inputted, firstinp, secondinp, win, played, _manual_history, _test_ran
    inputted, firstinp, secondinp, played, _manual_history = [], [], [], [], []
    win = 0
    _test_ran = False
    winrate_label.config(text="—")
    rounds_label.config(text="0 rounds")
    winorloselabel.config(text="")
    result_label.config(text="", bg=ACCENT, fg="black")
    history_canvas.delete("all")

def _refresh_dropdown():
    saved   = _scan_saved_datasets()
    options = [BUILTIN_907] + saved
    test_dropdown["values"] = options
    if test_dropdown.get() not in options:
        test_dropdown.set(BUILTIN_907)
    _update_dataset_info()

def _update_dataset_info():
    sel = test_dropdown.get()
    if sel == BUILTIN_907:
        dataset_info_label.config(text=f"{len(testsample)} entries")
    else:
        entries = _load_dataset(sel)
        dataset_info_label.config(text=f"{len(entries)} entries")

# Precompute dataset indices and counts
dataset_counts = defaultdict(int)
dataset_indices = defaultdict(list)
for i, val in enumerate(dataset):
    dataset_counts[val] += 1
    dataset_indices[val].append(i)

dataset_int = [int(x) for x in dataset]

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

def prepare_data(sequence, n_lags=2):
    arr = np.array(sequence)
    X = np.column_stack([arr[i : len(arr) - n_lags + i] for i in range(n_lags)])
    y = arr[n_lags:]
    return X, y

def _build_base_rf(data, n_lags=2):
    X, y = prepare_data(data, n_lags)
    model = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=None)
    model.fit(X, y)
    return model

_base_rf_first  = _build_base_rf(firstdataset)
_base_rf_second = _build_base_rf(seconddataset)
_base_rf_full   = _build_base_rf(dataset_int)

def predict_next_fast(base_rf, base_seq, user_seq, n_lags=2):
    combined = base_seq + user_seq
    if len(combined) < n_lags + 1: raise ValueError("short")
    last_values = np.array(combined[-n_lags:]).reshape(1, -1)
    pred_base = base_rf.predict(last_values)[0]
    if len(user_seq) >= n_lags + 1:
        X, y = prepare_data(user_seq, n_lags)
        if X.size > 0:
            user_rf = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=None)
            user_rf.fit(X, y)
            pred_user = user_rf.predict(last_values)[0]
            base_len = len(base_seq)
            w_base = base_len / (base_len + len(user_seq))
            w_user = len(user_seq) / (base_len + len(user_seq))
            return w_base * pred_base + w_user * pred_user
    return pred_base

def get_xgb_features(inp_array, window_size=10):
    arr = np.array(inp_array, dtype=float)
    N = len(arr)
    if N <= window_size:
        return np.empty((0, 6)), np.empty((0,)), np.empty((0, 6))
    windows = np.array([arr[i:i+window_size] for i in range(N - window_size)])
    y_train = arr[window_size:]
    means, stds, medians = np.mean(windows,axis=1), np.std(windows,axis=1), np.median(windows,axis=1)
    maxs,  mins  = np.max(windows,axis=1), np.min(windows,axis=1)
    ranges = maxs - mins
    X_train = np.column_stack((means, stds, medians, maxs, mins, ranges))
    ng = arr[-window_size:]
    X_pred = np.array([[np.mean(ng),np.std(ng),np.median(ng),np.max(ng),np.min(ng),np.max(ng)-np.min(ng)]])
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
        number   = int(key)
        distance = abs(number - target_number)
        confidence[key] += (10 * normal_pdf(distance, 0, 10)) * weight
    for key in confidence.keys():
        if int(key) == target_number:
            confidence[key] += 0.35 * weight

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
    mc = {state: dict(nexts) for state, nexts in base_mc.items()}
    seq = user_data
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
    try: nextfirstdiff = round(float(predict_next_fast(_base_rf_first, firstdataset, firstinp)))
    except ValueError: pass
    if nextfirstdiff == 10: nextseconddiff = 0
    else:
        try: nextseconddiff = round(float(predict_next_fast(_base_rf_second, seconddataset, secondinp)))
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

    try:
        mc = _mc_from_base(_base_mc_first, firstinp, 1)
        nextfirstdiff = int(predict_next_elementmark(mc, tuple(first_train[-1:])))
    except: pass
    if nextfirstdiff == 10: nextseconddiff = 0
    else:
        try:
            mc = _mc_from_base(_base_mc_second, secondinp, 1)
            nextseconddiff = int(predict_next_elementmark(mc, tuple(second_train[-1:])))
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
    if nextfirstdiff == 10: nextseconddiff = 0
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
    try:
        inputted_int = [int(x) for x in inputted]
        nextfirstdiff = round(float(predict_next_fast(_base_rf_full, dataset_int, inputted_int)))
    except: pass
    if nextfirstdiff is not None: othernormaldist(int(nextfirstdiff), 8)
    nextfirstdiff = None

    try:
        mc = _mc_from_base(_base_mc_full, inputted, 1)
        nextfirstdiff = int(predict_next_elementmark(mc, tuple(main_train[-1:])))
    except: pass
    if nextfirstdiff is not None: othernormaldist(int(nextfirstdiff), 4.6)
    nextfirstdiff = None
    try: nextfirstdiff = frequency2[inputted[-1]]
    except: pass
    if nextfirstdiff is not None: othernormaldist(int(nextfirstdiff), 4.7)
    return confidence

def main():
    global inputted, retro, temp, tempc, next_element, confidence, firstinp, secondinp
    next_element, difference = 0, 0
    confidence = differencepred()
    input_len  = len(inputted)
    base_add   = (len(dataset) + input_len) / 7500000

    for val, count in dataset_counts.items():
        if val in confidence:
            confidence[val] += count * base_add

    if input_len > 0:
        last_val = inputted[-1]
        for i in dataset_indices.get(last_val, []):
            j_limit = min(len(dataset) - i, 1000002)
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
            age_decay = 0.5 ** ((input_len - 1 - i) / 1000)
            retro = (i / input_len) * age_decay
            confidence[inputted[i]] += 0.7 * retro
            if inputted[i] == last_val:
                j_limit = min(input_len - i, 1000002)
                if j_limit > 2:
                    L = 1
                    while L < j_limit - 1 and L < input_len:
                        if inputted[i - L] == inputted[-1 - L]: L += 1
                        else: break
                    if L >= 2:
                        confidence[inputted[i + 1]] += (L * (L - 1) / 2) * 13 * retro
                    elif L == 1:
                        confidence[inputted[i + 1]] += 3.5 * retro

    if (len(inputted) >= 2) and (int(inputted[-2]) - int(inputted[-1]) in {1,2,3,5,10,20,-1,-2,-3,-5,-10,-20}):
        next_element = int(inputted[-1]) + (int(inputted[-1]) - int(inputted[-2]))
        if 0 <= next_element <= 9: next_element = f"0{next_element}"
        if 0 <= int(next_element) <= 100: confidence[str(next_element)] += 12

    if (len(inputted) >= 3) and (inputted[-1] != inputted[-2]) and (int(inputted[-1]) - int(inputted[-2])) == (int(inputted[-2]) - int(inputted[-3])):
        difference   = int(inputted[-1]) - int(inputted[-2])
        next_element = int(inputted[-1]) + difference
        if 0 <= next_element <= 9: next_element = f"0{next_element}"
        if 0 <= int(next_element) <= 100: confidence[str(next_element)] += 30

    try:
        if (len(inputted) >= 2) and ((int(inputted[-2])/int(inputted[-1])) in {2, 0.5}):
            next_element = int(int(inputted[-1]) * (int(inputted[-1]) / int(inputted[-2])))
            if 0 <= int(next_element) <= 9:  next_element = f"0{next_element}"
            if 0 <= int(next_element) <= 100: confidence[str(next_element)] += 18
    except: pass

    try:
        ratios = [int(inputted[i]) / int(inputted[i-1]) for i in range(len(inputted)-3, len(inputted))]
        if all(r == ratios[0] for r in ratios):
            next_element = int(int(inputted[-1]) * ratios[0])
            if 0 <= next_element <= 9:  next_element = f"0{next_element}"
            if 0 <= int(next_element) <= 100: confidence[str(next_element)] += 30
    except: pass

    try:
        if len(inputted) >= 5 and (int(inputted[-1])-int(inputted[-3])) == (int(inputted[-3])-int(inputted[-5])):
            next_element = int(inputted[-2]) + (int(inputted[-2]) - int(inputted[-4]))
            if 0 <= next_element <= 9:  next_element = f"0{next_element}"
            if 0 <= int(next_element) <= 100: confidence[str(next_element)] += 30
    except: pass

    try:
        if len(inputted) >= 3 and int(inputted[-1]) == int(inputted[-2]) + int(inputted[-3]):
            next_element = int(inputted[-1]) + int(inputted[-2])
            if 0 <= next_element <= 9:  next_element = f"0{next_element}"
            if 0 <= int(next_element) <= 100: confidence[str(next_element)] += 20
    except: pass

    try:
        if len(inputted) >= 3 and inputted[-1] == inputted[-3] and inputted[-1] != inputted[-2]:
            next_element = inputted[-2]
            if 0 <= int(next_element) <= 100: confidence[str(next_element)] += 50
    except: pass
    try:
        if len(inputted) >= 5 and inputted[-1] == inputted[-3] == inputted[-5] and inputted[-2] == inputted[-4] and inputted[-1] != inputted[-2]:
            next_element = inputted[-2]
            if 0 <= int(next_element) <= 100: confidence[str(next_element)] += 100
    except: pass

    try:
        if len(inputted) >= 4:
            d1 = int(inputted[-1])-int(inputted[-2]); d2 = int(inputted[-2])-int(inputted[-3]); d3 = int(inputted[-3])-int(inputted[-4])
            if d1 == d3 and d1 != d2:
                next_element = int(inputted[-1]) + d2
                if 0 <= next_element <= 9:  next_element = f"0{next_element}"
                if 0 <= int(next_element) <= 100: confidence[str(next_element)] += 20
    except: pass

    try:
        if len(inputted) >= 4:
            d1 = int(inputted[-1])-int(inputted[-2]); d2 = int(inputted[-2])-int(inputted[-3]); d3 = int(inputted[-3])-int(inputted[-4])
            if (d1-d2) == (d2-d3) and d1 != d2:
                next_element = int(inputted[-1]) + d1 + (d1-d2)
                if 0 <= next_element <= 9:  next_element = f"0{next_element}"
                if 0 <= int(next_element) <= 100: confidence[str(next_element)] += 15
    except: pass

    try:
        if input_len >= 2:
            last_1 = inputted[-1]; freq_1 = defaultdict(int)
            for i in range(input_len - 1):
                if inputted[i] == last_1: freq_1[inputted[i+1]] += 1
            for nv, c in freq_1.items():
                if c >= 2: confidence[nv] += (c ** 3) * 1.5 #Increased from 2.5
        if input_len >= 3:
            lv1, lv2 = inputted[-1], inputted[-2]; freq_2 = defaultdict(int)
            for i in range(input_len - 2):
                if inputted[i] == lv2 and inputted[i+1] == lv1: freq_2[inputted[i+2]] += 1
            for nv, c in freq_2.items():
                if c >= 2: confidence[nv] += (c ** 3.5) * 2
    except: pass

    if len(inputted) == 0: return "37"
    if max(confidence.values()) == 0.0: return inputted[-1]
    return max(confidence, key=confidence.get)


# ── Manual history bar ────────────────────────────────────────────────────────
_manual_history = []

def _redraw_manual_history():
    history_canvas.delete("all")
    step = 9
    cw   = history_canvas.winfo_width()
    vis  = cw // step if cw > 1 else 100
    for i, ok in enumerate(_manual_history[-vis:]):
        x0 = i * step
        history_canvas.create_rectangle(x0, 0, x0+7, 16, fill="#22cc44" if ok else "#dd2222", outline="")

def numinput(event=None):
    global win, confidence, played, timerup, inputted, _create_mode, _current_entries, _manual_history
    try:
        if timerup and not _create_mode: raise ValueError
        input_text = entry.get()
        entry.delete(0, "end")
        result_label.config(text="")
        if (0 <= int(input_text) <= 100) and ((((input_text[0] not in {"0"," "}) == (0 <= int(input_text)) <= 100)) or input_text == "0"):
            num_str = input_text if int(input_text) >= 10 else f"0{int(input_text)}"
            if _create_mode:
                _current_entries.append(num_str)
                create_count_label.config(text=f"● REC  {len(_current_entries)} entries")
                result_label.config(text=f"+{num_str}", bg=BG, fg="#22cc44")
                result_label.after(400, lambda: result_label.config(text="", fg="black"))
            else:
                if _test_ran:
                    _reset_session()
                returned = main()
                inputted.append(input_text)
                if 0 <= int(inputted[-1]) <= 9: inputted[-1] = f"0{inputted[-1]}"
                played.insert(0, returned); 
                if len(played) >= 4: played.pop(-1)
                correct = inputted[-1] == returned
                _manual_history.append(correct)
                if correct:
                    pygame.mixer.music.load(correctsfx); pygame.mixer.music.play()
                    win += 1
                    winorloselabel.config(text="Bot Wins", fg="#22cc44")
                    result_label.config(text=returned if show_guess_var.get() else "✓", bg="lawn green", fg="black")
                else:
                    pygame.mixer.music.load(wrongsfx); pygame.mixer.music.play()
                    winorloselabel.config(text="Bot Lost", fg="#dd2222")
                    result_label.config(text=returned if show_guess_var.get() else "✗", bg="red2", fg="white")
                rate = win / len(inputted) * 100
                winrate_label.config(text=f"{rate:.3f}%")
                rounds_label.config(text=f"{len(inputted)} rounds")
                result_label.after(200, lambda: result_label.config(bg=ACCENT, fg="black"))
                _redraw_manual_history()
                confidence_str = ""
                for key, value in confidence.items():
                    confidence_str += f"{key}: {value:.2f}, "
                    if int(key) % 6 == 0: confidence_str += "\n"
                confidencelabel.config(text=f"Confidence levels for prior number:\n{confidence_str}\n(don't use these to cheat weirdo)")
        else: raise ValueError
    except ValueError:
        result_label.config(text="?", bg="#ffcccc", fg="#cc0000")
        result_label.after(500, lambda: result_label.config(bg=ACCENT, fg="black", text=""))
    entry.focus_set()


def _prompt_save_dataset():
    global _current_entries
    if not _current_entries: return
    dialog = tk.Toplevel(root)
    dialog.title("Save Dataset")
    dialog.configure(bg=BG)
    dialog.resizable(False, False)
    dialog.grab_set()
    root.update_idletasks()
    x = root.winfo_x() + root.winfo_width()  // 2 - 210
    y = root.winfo_y() + root.winfo_height() // 2 - 90
    dialog.geometry(f"420x180+{x}+{y}")

    tk.Label(dialog, text=f"Name this dataset  ({len(_current_entries)} entries)",
             font=("Helvetica", 14, "bold"), bg=BG).pack(pady=(20, 6))
    name_var   = tk.StringVar(value="my_dataset")
    name_entry = tk.Entry(dialog, textvariable=name_var, font=("Helvetica", 15), width=22,
                          relief="solid", bd=1)
    name_entry.pack()
    name_entry.focus_set()
    name_entry.select_range(0, "end")

    def do_save(evt=None):
        global _current_entries  # ← add this
        raw  = name_var.get().strip()
        if not raw: return
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in raw)
        _save_dataset(safe, _current_entries)
        _current_entries = []
        _refresh_dropdown()
        test_dropdown.set(safe)
        _update_dataset_info()
        dialog.destroy()

    def do_discard():
        global _current_entries  # ← add this
        _current_entries = []
        dialog.destroy()

    bf = tk.Frame(dialog, bg=BG); bf.pack(pady=14)
    tk.Button(bf, text="Save", font=("Helvetica", 12, "bold"), bg="#22cc44", padx=18,
              relief="flat", command=do_save).pack(side="left", padx=10)
    tk.Button(bf, text="Discard", font=("Helvetica", 12), bg="#dd2222", fg="white",
              padx=14, relief="flat", command=do_discard).pack(side="left", padx=10)
    name_entry.bind("<Return>", do_save)
    dialog.wait_window()


def toggle_create_mode():
    global _create_mode, _current_entries
    _create_mode = not _create_mode
    if _create_mode:
        _current_entries = []
        create_btn.config(text="⏹  Stop Recording", bg="#dd3333", fg="white", relief="flat")
        create_count_label.config(text="● REC  0 entries", fg="#dd3333")
        result_label.config(text="REC", bg="#dd3333", fg="white")
        result_label.after(800, lambda: result_label.config(text="", bg=ACCENT, fg="black"))
    else:
        create_btn.config(text="⏺  Create Dataset", bg=BTN_BG, fg=BTN_FG, relief="flat")
        create_count_label.config(text="", fg="#555555")
        if _current_entries:
            root.after(100, _prompt_save_dataset)


def run_selected_test(event=None):
    global _test_running
    if _test_running: return
    sel = test_dropdown.get()
    if sel == BUILTIN_907:
        sample, label = testsample, "907"
    else:
        sample = _load_dataset(sel)
        if len(sample) < 2:
            progress_label.config(text=f"'{sel}' is empty or missing.")
            return
        label = sel
    _reset_session()
    progress_bar["value"] = 0
    streak_label.config(text="Streak  0   Best  0   Worst loss  0")
    last_label.config(text="Waiting...", bg=BG, fg="#333333")
    _test_running = True
    run_test_btn.config(state="disabled")
    _run_auto_test(sample, label)


def _run_auto_test(sample, label):
    global win, inputted, _test_running
    ts = {"win":0,"streak":0,"best_streak":0,"lose_streak":0,"worst_streak":0,
          "history":[],"start_time":time.time(),"last_actual":"—","last_guess":"—","last_correct":None}
    progress_bar["maximum"] = len(sample)

    def ui(state, idx):
        total   = len(sample)
        rate    = state["win"] / idx * 100 if idx else 0
        elapsed = time.time() - state["start_time"]
        rps     = idx / elapsed if elapsed > 0 else 0
        eta     = (total - idx) / rps if rps > 0 else 0
        progress_bar["value"] = idx
        progress_label.config(text=f"[{label}]  {idx}/{total}  ·  {rate:.2f}%  ·  {rps:.1f}/s  ·  ETA {eta:.0f}s")
        streak_label.config(text=f"Streak  {state['streak']}   Best  {state['best_streak']}   Worst loss  {state['worst_streak']}")
        if show_guess_var.get():
            last_label.config(text=f"Actual: {state['last_actual']}    AI: {state['last_guess']}",
                              bg="lawn green" if state["last_correct"] else "red2",
                              fg="black" if state["last_correct"] else "white")
        else:
            last_label.config(text="✓  correct" if state["last_correct"] else "✗  wrong",
                              bg="lawn green" if state["last_correct"] else "red2",
                              fg="black" if state["last_correct"] else "white")
        history_canvas.delete("all")
        cw  = history_canvas.winfo_width()
        vis = cw // 9 if cw > 1 else 100
        for i, ok in enumerate(state["history"][-vis:]):
            x0 = i * 9
            history_canvas.create_rectangle(x0, 0, x0+7, 16, fill="#22cc44" if ok else "#dd2222", outline="")
        winrate_label.config(text=f"{rate:.3f}%")
        rounds_label.config(text=f"{idx} rounds")

    def run():
        global win, inputted, _test_running
        for idx, txt in enumerate(sample, 1):
            ret = main(); inputted.append(txt)
            ok  = txt == ret
            if ok:
                win += 1; ts["win"] += 1; ts["streak"] += 1; ts["lose_streak"] = 0
                ts["best_streak"] = max(ts["best_streak"], ts["streak"])
            else:
                ts["lose_streak"] += 1; ts["streak"] = 0
                ts["worst_streak"] = max(ts["worst_streak"], ts["lose_streak"])
            ts["history"].append(ok); ts["last_actual"] = txt; ts["last_guess"] = ret; ts["last_correct"] = ok
            root.after(0, ui, dict(ts), idx)

        def finish():
            global _test_running, _test_ran
            el   = time.time() - ts["start_time"]
            rate = ts["win"] / len(sample) * 100
            progress_label.config(text=f"[{label}]  Done · {len(sample)} rounds · {el:.1f}s · Final: {rate:.3f}%")
            last_label.config(text="Complete", bg=BG, fg="#333333")
            run_test_btn.config(state="normal")
            _test_running = False
            _test_ran = True
        root.after(0, finish)

    threading.Thread(target=run, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# UI BUILD
# ─────────────────────────────────────────────────────────────────────────────
_test_running = False
_test_ran = False
pygame.mixer.init()
timerup = False

BG      = "#aee8e8"   # pale turquoise
PANEL   = "#c8f0f0"   # slightly lighter panel bg
ACCENT  = "#87ceeb"   # skyblue1
BTN_BG  = "#4a7fa5"
BTN_FG  = "white"
DARK    = "#1a2d3d"

root = tk.Tk()
root.title("Number Predictor Thing")
root.configure(bg=BG)
root.attributes("-fullscreen", True)

style = ttk.Style()
style.theme_use("clam")
style.configure("TProgressbar", troughcolor=PANEL, background="#4a7fa5", thickness=10)
style.configure("TCombobox", fieldbackground="white", background="white", font=("Helvetica", 13))

def toggle_fullscreen(event=None):
    root.attributes("-fullscreen", False)
root.bind("<Escape>", toggle_fullscreen)

# ── Root grid: left panel (col 0) + right sidebar (col 1) ────────────────────
root.grid_columnconfigure(0, weight=3)
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(0, weight=0)   # title + input
root.grid_rowconfigure(1, weight=0)   # controls
root.grid_rowconfigure(2, weight=1)   # stats (expandable)
root.grid_rowconfigure(3, weight=0)   # result readout

# ── Title + input ─────────────────────────────────────────────────────────────
top_frame = tk.Frame(root, bg=BG)
top_frame.grid(row=0, column=0, columnspan=1, sticky="ew", padx=30, pady=(18, 6))

tk.Label(top_frame, text="Number Predictor Thing",
         font=("Helvetica", 38, "bold"), bg="white",
         padx=20, pady=6, relief="flat").pack(fill="x")

entry = tk.Entry(top_frame, font=("Helvetica", 28), relief="solid", bd=1,
                 justify="center")
entry.pack(fill="x", pady=(10, 0), ipady=4)
entry.bind("<Return>", numinput)
entry.focus_set()

# ── Controls panel ────────────────────────────────────────────────────────────
ctrl_frame = tk.Frame(root, bg=PANEL, bd=0, relief="flat")
ctrl_frame.grid(row=1, column=0, sticky="ew", padx=30, pady=(8, 4))
ctrl_frame.grid_columnconfigure(0, weight=0)

img_button    = tk.PhotoImage(file=checknumberbutton)
img_907button = tk.PhotoImage(file=standardizedtestbutton)

# ── Row A: Check + Run + dropdown only ───────────────────────────────────────
rowA = tk.Frame(ctrl_frame, bg=PANEL)
rowA.pack(fill="x", padx=12, pady=(10, 4))

check_button = tk.Button(rowA, image=img_button, borderwidth=0,
                         bg=PANEL, activebackground=PANEL)
check_button.pack(side="left", padx=(0, 8))

run_test_btn = tk.Button(rowA, image=img_907button, borderwidth=0,
                         bg=PANEL, activebackground=PANEL,
                         command=run_selected_test)
run_test_btn.pack(side="left", padx=(0, 12))

test_dropdown = ttk.Combobox(rowA, state="readonly", font=("Helvetica", 13), width=18)
test_dropdown.pack(side="left", padx=(0, 8))
test_dropdown.bind("<<ComboboxSelected>>", lambda e: _update_dataset_info())

# ── Row C: Create dataset + entry count + dataset info + show guess ───────────
rowC = tk.Frame(ctrl_frame, bg=PANEL)
rowC.pack(fill="x", padx=12, pady=(0, 10))

create_btn = tk.Button(rowC, text="⏺  Create Dataset",
                       font=("Helvetica", 12, "bold"),
                       bg=BTN_BG, fg=BTN_FG,
                       padx=14, pady=6, relief="flat",
                       activebackground="#3a6f95",
                       command=toggle_create_mode)
create_btn.pack(side="left", padx=(0, 14))

create_count_label = tk.Label(rowC, text="", font=("Helvetica", 12),
                              bg=PANEL, fg="#555555")
create_count_label.pack(side="left", padx=(0, 16))

dataset_info_label = tk.Label(rowC, text="", font=("Helvetica", 12),
                              bg=PANEL, fg="#444444")
dataset_info_label.pack(side="left", padx=(0, 16))

show_guess_var = tk.BooleanVar(value=True)
tk.Checkbutton(rowC, text="Show AI guess", variable=show_guess_var,
               font=("Helvetica", 13), bg=PANEL,
               activebackground=PANEL).pack(side="left")

# ── Stats panel ───────────────────────────────────────────────────────────────
stats_frame = tk.Frame(root, bg=BG)
stats_frame.grid(row=2, column=0, sticky="nsew", padx=30, pady=4)
stats_frame.grid_columnconfigure(0, weight=1)

progress_bar = ttk.Progressbar(stats_frame, orient="horizontal",
                               mode="determinate", maximum=len(testsample))
progress_bar.pack(fill="x", pady=(4, 2))

progress_label = tk.Label(stats_frame,
                           text="Select a dataset above and press Run to start a test",
                           font=("Helvetica", 12), bg=BG, fg="#333333")
progress_label.pack()

streak_label = tk.Label(stats_frame,
                        text="Streak  0    Best  0    Worst loss  0",
                        font=("Helvetica", 12, "bold"), bg=BG, fg=DARK)
streak_label.pack(pady=2)

last_label = tk.Label(stats_frame, text="Actual: —    AI: —",
                      font=("Helvetica", 13, "bold"), bg=BG, fg=DARK, width=44)
last_label.pack(pady=2)

history_canvas = tk.Canvas(stats_frame, height=16, bg=BG, highlightthickness=0)
history_canvas.pack(fill="x", pady=(2, 4))

# ── Result readout (bottom of left column) ────────────────────────────────────
result_frame = tk.Frame(root, bg=BG)
result_frame.grid(row=3, column=0, sticky="ew", padx=30, pady=(4, 20))

result_label = tk.Label(result_frame, text="",
                        font=("Helvetica", 52, "bold"),
                        bg=ACCENT, fg="black",
                        relief="flat")
result_label.pack(side="left", padx=(0, 30))

readout_right = tk.Frame(result_frame, bg=BG)
readout_right.pack(side="left")

winorloselabel = tk.Label(readout_right, text="",
                          font=("Helvetica", 26, "bold"),
                          bg=BG, fg=DARK)
winorloselabel.pack(anchor="w")

winrate_label = tk.Label(readout_right, text="—",
                         font=("Helvetica", 30, "bold"),
                         bg=BG, fg=DARK)
winrate_label.pack(anchor="w")

rounds_label = tk.Label(readout_right, text="0 rounds",
                        font=("Helvetica", 15),
                        bg=BG, fg="#555555")
rounds_label.pack(anchor="w")

# ── Right sidebar: confidence levels ─────────────────────────────────────────
right_frame = tk.Frame(root, bg=BG)
right_frame.grid(row=0, column=1, rowspan=4, sticky="nsew", padx=(0, 20), pady=20)

confidenceinit = {str(i).zfill(2): 0 for i in range(0, 101)}
confidence_str = ""
for key, value in confidenceinit.items():
    confidence_str += f"{key}: {value:.2f}, "
    if int(key) % 6 == 0: confidence_str += "\n"

confidencelabel = tk.Label(right_frame,
                           text=f"Confidence levels for prior number:\n{confidence_str}\n(don't use these to cheat weirdo)",
                           fg=DARK, bg=BG, font=("Helvetica", 10), justify="left")
confidencelabel.pack(pady=10, anchor="nw")

check_button.bind("<Button-1>", numinput)
run_test_btn.bind("<Button-1>", run_selected_test)

root.after(150, _refresh_dropdown)

if __name__ == "__main__":
    root.mainloop()
