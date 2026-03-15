import os
import pygame
import threading
import tkinter as tk
import numpy as np
from collections import deque
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from data import dataset, firstdataset, seconddataset, testsample, frequency, frequency2
import warnings
warnings.filterwarnings("ignore")

# File paths — resolved relative to this script so they work regardless of cwd
_dir = os.path.dirname(os.path.abspath(__file__))
assets = os.path.join(_dir, "assets") + os.sep
checknumberbutton = assets + os.path.join("images", "check.png")
standardizedtestbutton = assets + os.path.join("images", "run907.png")
correctsfx = assets + os.path.join("audios", "correct.mp3")
wrongsfx = assets + os.path.join("audios", "wrong.mp3")

# Global state
inputted = []
firstinp = deque(maxlen=500)   # bounded to avoid unbounded memory growth
secondinp = deque(maxlen=500)
temp, tempc = [], []
win = 0
played = []
confidence = {str(i).zfill(2): 0 for i in range(0, 101)}

# Model / chain caches — rebuilt only when new data arrives
_rf_first_cache = {"model": None, "train_len": -1}
_rf_second_cache = {"model": None, "train_len": -1}
_xgb_first_cache = {"model": None, "train_len": -1}
_xgb_second_cache = {"model": None, "train_len": -1}
_markov_full_cache = {"chain": None, "train_len": -1}
_markov_first_cache = {"chain": None, "train_len": -1}
_markov_second_cache = {"chain": None, "train_len": -1}

# ----- Machine Learning / Prediction Functions -----

def prepare_data(sequence, n_lags=5):
    seq = [float(x) for x in sequence]
    X, y = [], []
    for i in range(len(seq) - n_lags):
        X.append(seq[i:i + n_lags])
        y.append(seq[i + n_lags])
    return np.array(X), np.array(y)

def predict_next(sequence, n_lags=5):
    if len(sequence) < n_lags + 1:
        raise ValueError("short")
    X, y = prepare_data(sequence, n_lags)
    if X.size == 0 or y.size == 0:
        raise ValueError("short")
    model = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)
    model.fit(X, y)
    last_values = np.array(sequence[-n_lags:]).reshape(1, -1)
    return model.predict(last_values)[0]   # [0] — extract scalar from array

def normal_pdf(x, mean, sigma):
    factor = 1 / (sigma * (2 * 3.141592653589793) ** 0.5)
    exponent = -((x - mean) ** 2) / (2 * sigma ** 2)
    return factor * (2.718281828459045 ** exponent)

def normaldist(target_first_digit, target_second_digit, weight):
    global confidence
    for key in confidence.keys():
        if key != "100":
            first_digit = int(key[0])
        else:
            first_digit = 10
        distance = abs(first_digit - target_first_digit)
        confidence[key] += (normal_pdf(distance, 0, 2)) * weight
        second_digit = int(key[1])
        distance = abs(second_digit - target_second_digit)
        confidence[key] += (2 - (distance / 10)) * weight
        if int(key) == (target_first_digit * 10 + target_second_digit):
            confidence[key] += 1 * weight
        if int(key[1]) == target_second_digit:
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
    markov_chain = {}
    for i in range(len(data) - k):
        current_state = tuple(data[i:i + k])
        next_state = data[i + k]
        if current_state not in markov_chain:
            markov_chain[current_state] = {}
        markov_chain[current_state].setdefault(next_state, 0)
        markov_chain[current_state][next_state] += 1
    return markov_chain

def update_markov_chain(chain, data, k):
    """Incrementally add the last transition to an existing chain."""
    if len(data) < k + 1:
        return chain
    current_state = tuple(data[-(k + 1):-1])
    next_state = data[-1]
    if current_state not in chain:
        chain[current_state] = {}
    chain[current_state].setdefault(next_state, 0)
    chain[current_state][next_state] += 1
    return chain

def predict_next_elementmark(markov_chain, current_state):
    while current_state not in markov_chain and len(current_state) > 1:
        current_state = current_state[1:]
    if current_state in markov_chain:
        transitions = markov_chain[current_state]
        total_count = sum(transitions.values())
        if total_count > 0:
            probabilities = {state: count / total_count for state, count in transitions.items()}
            return max(probabilities, key=probabilities.get)
    overall_transitions = {}
    for state, transitions in markov_chain.items():
        for next_state, count in transitions.items():
            overall_transitions[next_state] = overall_transitions.get(next_state, 0) + count
    if overall_transitions:
        total_count = sum(overall_transitions.values())
        if total_count > 0:
            probabilities = {state: count / total_count for state, count in overall_transitions.items()}
            return max(probabilities, key=probabilities.get)
    return None

def _get_xgb_model(cache, train, window_size=10):
    """Return cached XGB model, rebuilding only when training data has grown."""
    if cache["train_len"] == len(train):
        return cache["model"]
    X_train, y_train = [], []
    for i in range(len(train) - window_size):
        group = np.array(train[i:i + window_size], dtype=float)
        X_train.append([
            np.mean(group), np.std(group), np.median(group),
            np.max(group), np.min(group), np.max(group) - np.min(group)
        ])
        y_train.append(float(train[i + window_size]))
    if not X_train:
        cache["model"] = None
        cache["train_len"] = len(train)
        return None
    model = xgb.XGBRegressor(
        n_estimators=35, max_depth=10,
        learning_rate=0.11, objective='reg:squarederror'
    )
    model.fit(np.array(X_train), np.array(y_train))
    cache["model"] = model
    cache["train_len"] = len(train)
    return model

def _xgb_predict(model, sequence, window_size=10):
    group = np.array(list(sequence[-window_size:]), dtype=float)
    features = np.array([[
        np.mean(group), np.std(group), np.median(group),
        np.max(group), np.min(group), np.max(group) - np.min(group)
    ]])
    return int(model.predict(features)[0])   # [0] — extract scalar

def differencepred():
    global nextfirstdiff, nextseconddiff, confidence, firstinp, secondinp, inputted
    global _rf_first_cache, _rf_second_cache
    global _markov_full_cache, _markov_first_cache, _markov_second_cache

    confidence = {str(i).zfill(2): 0 for i in range(0, 101)}
    if len(inputted) == 0:
        return confidence

    try:
        if inputted[-1] == "100":
            firstinp.append(10)
        else:
            firstinp.append(int(inputted[-1][0]))
        secondinp.append(int(inputted[-1][1]))
    except:
        pass

    first_list = list(firstdataset) + list(firstinp)
    second_list = list(seconddataset) + list(secondinp)
    full_list = list(dataset) + list(inputted)

    nextfirstdiff, nextseconddiff = None, None

    # --- RandomForest on digit sequences ---
    try:
        nextfirstdiff = round(float(predict_next(first_list)))
    except ValueError:
        pass
    if nextfirstdiff == 10:
        nextseconddiff = 0
    else:
        try:
            nextseconddiff = round(float(predict_next(second_list)))
        except ValueError:
            pass
    if nextfirstdiff is not None and nextseconddiff is not None:
        normaldist(nextfirstdiff, nextseconddiff, 1)

    nextfirstdiff, nextseconddiff = None, None

    # --- Frequency table lookup ---
    try:
        nextfirstdiff = frequency[inputted[-1]][0]
        nextseconddiff = 0 if nextfirstdiff == 10 else frequency[inputted[-1]][1]
        normaldist(nextfirstdiff, nextseconddiff, 1.1)
    except:
        pass

    nextfirstdiff, nextseconddiff = None, None

    # --- Markov on digit sequences (cached) ---
    if _markov_first_cache["train_len"] != len(first_list):
        _markov_first_cache["chain"] = build_markov_chain(first_list, 1)
        _markov_first_cache["train_len"] = len(first_list)
    if _markov_second_cache["train_len"] != len(second_list):
        _markov_second_cache["chain"] = build_markov_chain(second_list, 1)
        _markov_second_cache["train_len"] = len(second_list)

    try:
        current_state = tuple(first_list[-1:])
        nextfirstdiff = int(predict_next_elementmark(_markov_first_cache["chain"], current_state))
    except:
        pass
    if nextfirstdiff == 10:
        nextseconddiff = 0
    else:
        try:
            current_state = tuple(second_list[-1:])
            nextseconddiff = int(predict_next_elementmark(_markov_second_cache["chain"], current_state))
        except:
            pass
    if nextfirstdiff is not None and nextseconddiff is not None:
        normaldist(nextfirstdiff, nextseconddiff, 1.7)

    nextfirstdiff, nextseconddiff = None, None

    # --- XGBoost on digit sequences (cached) ---
    window_size = 10
    if len(first_list) > window_size:
        model = _get_xgb_model(_xgb_first_cache, first_list, window_size)
        if model:
            try:
                nextfirstdiff = _xgb_predict(model, first_list, window_size)
            except:
                pass
    if nextfirstdiff == 100:
        nextseconddiff = 0
    elif len(second_list) > window_size:
        model = _get_xgb_model(_xgb_second_cache, second_list, window_size)
        if model:
            try:
                nextseconddiff = _xgb_predict(model, second_list, window_size)
            except:
                pass
    if nextfirstdiff is not None and nextseconddiff is not None:
        normaldist(nextfirstdiff, nextseconddiff, 1.1)

    # --- RandomForest on full sequence ---
    nextfirstdiff = None
    try:
        nextfirstdiff = round(float(predict_next(full_list)))
    except:
        pass
    if nextfirstdiff is not None:
        othernormaldist(int(nextfirstdiff), 8)   # fixed: was guarded by nextseconddiff

    # --- Markov on full sequence (cached) ---
    nextfirstdiff = None
    if _markov_full_cache["train_len"] != len(full_list):
        _markov_full_cache["chain"] = build_markov_chain(full_list, 1)
        _markov_full_cache["train_len"] = len(full_list)
    try:
        current_state = tuple(full_list[-1:])
        nextfirstdiff = int(predict_next_elementmark(_markov_full_cache["chain"], current_state))
    except:
        pass
    if nextfirstdiff is not None:
        othernormaldist(int(nextfirstdiff), 4.6)

    # --- Frequency2 lookup ---
    nextfirstdiff = None
    try:
        nextfirstdiff = frequency2[inputted[-1]]
    except:
        pass
    if nextfirstdiff is not None:
        othernormaldist(int(nextfirstdiff), 4.8)

    return confidence


def main():
    global inputted, temp, tempc, next_element, confidence, firstinp, secondinp
    next_element, difference = 0, 0
    confidence = differencepred()

    # --- Dataset pattern matching ---
    for i in range(len(dataset)):
        confidence[dataset[i]] += (20609 + len(inputted)) / 7500000
        try:
            for j in range(2, min(1002, len(dataset) - i)):   # cap at 1002, not 1000002
                temp, tempc = [], []
                for k in range(j):
                    temp.insert(0, dataset[i - k])
                    tempc.insert(0, inputted[-1 - k])
                if temp == tempc:
                    confidence[dataset[i + 1]] += (j - 1) * 4.6
                else:
                    break
        except:
            pass

    # --- User history pattern matching ---
    for i in range(len(inputted)):
        retro = i / len(inputted)
        confidence[inputted[i]] += 0.7 * retro
        for j in range(2, min(1002, len(inputted) - i)):    # cap at 1002
            temp, tempc = [], []
            for k in range(j):
                temp.insert(0, inputted[i - k])
                tempc.insert(0, inputted[-1 - k])
            if temp == tempc:
                confidence[inputted[i + 1]] += (j - 1) * 10.9 * retro
            else:
                break

    # --- Arithmetic sequence detection ---
    if (len(inputted) >= 2) and (int(inputted[-2]) - int(inputted[-1]) in {1, 2, 3, 5, 10, 20, -1, -2, -3, -5, -10, -20}):
        next_element = int(inputted[-1]) + (int(inputted[-1]) - int(inputted[-2]))
        if 0 <= next_element <= 9:
            next_element = f"0{next_element}"
        if 0 <= int(next_element) <= 100:
            confidence[str(next_element)] += 10

    if (len(inputted) >= 3) and (inputted[-1] != inputted[-2]) and (int(inputted[-1]) - int(inputted[-2])) == (int(inputted[-2]) - int(inputted[-3])):
        difference = int(inputted[-1]) - int(inputted[-2])
        next_element = int(inputted[-1]) + difference
        if 0 <= next_element <= 9:
            next_element = f"0{next_element}"
        if 0 <= int(next_element) <= 100:
            confidence[str(next_element)] += 30

    # --- Geometric sequence detection ---
    try:
        if (len(inputted) >= 2) and ((int(inputted[-2]) / int(inputted[-1])) in {2, 0.5}):
            next_element = int(int(inputted[-1]) * (int(inputted[-1]) / int(inputted[-2])))
            if 0 <= int(next_element) <= 9:
                next_element = f"0{next_element}"
            if 0 <= int(next_element) <= 100:
                confidence[str(next_element)] += 7
    except:
        pass

    try:
        ratios = [int(inputted[i]) / int(inputted[i - 1]) for i in range(len(inputted) - 3, len(inputted))]
        if all(ratio == ratios[0] for ratio in ratios):
            next_element = int((int(inputted[-1])) * ratios[0])
            if 0 <= next_element <= 9:
                next_element = f"0{next_element}"
            if 0 <= int(next_element) <= 100:
                confidence[str(next_element)] += 30
    except:
        pass

    if len(inputted) == 0:
        return "37"

    try:
        if (inputted[-1] == played[1]) and (inputted[-2] == played[2]):
            return played[0]
    except:
        pass

    if max(confidence.values()) == 0.0:   # fixed: was comparing items() tuple to 0.0
        return inputted[-1]

    return max(confidence, key=confidence.get)   # fixed: was using inverted dict (lost ties)


# ----- Tkinter UI Functions -----

def start_ai_thread(input_text):
    def run():
        returned = main()
        root.after(0, update_ui_after_ai, input_text, returned)
    threading.Thread(target=run, daemon=True).start()

def update_ui_after_ai(input_text, returned):
    global win, confidence, confidencelabel, played, inputted
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
    botplayedlabel.config(text=f"AI Win Rate: {(win / len(inputted) * 100):.3f}%\nRounds Played: {len(inputted)}")
    confidence_str = ""
    for key, value in confidence.items():
        confidence_str += f"{key}: {value:.2f}, "
        if int(key) % 6 == 0:
            confidence_str += "\n"
    confidencelabel.config(
        text=f"Confidence levels for prior number:\n{confidence_str}\n(don't use these to cheat weirdo)",
        fg='black', bg="pale turquoise"
    )
    result_label.after(200, lambda: result_label.config(bg="skyblue1"))
    entry.focus_set()

def numinput(event=None):
    global win, confidence, confidencelabel, played, inputted
    try:
        if len(inputted) >= 500:
            raise ValueError
        input_text = entry.get().strip()
        entry.delete(0, "end")
        result_label.config(text="            ")
        # Clean validation: must be a digit string in range 0–100
        if not input_text.isdigit():
            raise ValueError
        val = int(input_text)
        if not (0 <= val <= 100):
            raise ValueError
        # Reject leading zeros (except "0" itself)
        if len(input_text) > 1 and input_text[0] == "0":
            raise ValueError
        start_ai_thread(input_text)
    except ValueError:
        result_label.config(text="poopy number", bg="skyblue1")
    entry.focus_set()

def autonuminput(event=None):
    global win, confidence, confidencelabel, inputted, firstinp, secondinp
    result_label.config(text="calculating")
    for input_text in testsample:
        returned = main()
        inputted.append(input_text)
        if input_text == returned:
            win += 1
        print(f"actual answer: {input_text} AI winrate {(win / len(inputted) * 100):.3f}% Rounds played {len(inputted)}/907")
    botplayedlabel.config(text=f"AI Win Rate: {(win / len(inputted) * 100):.3f}%\nRounds Played: {len(inputted)}")
    confidence_str = ""
    for key, value in confidence.items():
        confidence_str += f"{key}: {value:.2f}, "
        if int(key) % 6 == 0:
            confidence_str += "\n"
    result_label.config(text=" Done ")
    confidencelabel.config(
        text=f"Confidence levels for prior number:\n{confidence_str}\n(don't use these to cheat weirdo)",
        fg='black', bg="pale turquoise"
    )

# ----- UI Initialization -----
pygame.mixer.init()

root = tk.Tk()
root.title("Number Predictor Thing")
root.configure(bg="pale turquoise")
root.geometry("1280x620")
root.attributes("-fullscreen", True)

def toggle_fullscreen(event=None):
    root.attributes("-fullscreen", False)

root.bind("<Escape>", toggle_fullscreen)

# Layout frames
top_frame = tk.Frame(root, bg="pale turquoise")
top_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=10)
middle_frame = tk.Frame(root, bg="pale turquoise")
middle_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
bottom_frame = tk.Frame(root, bg="pale turquoise")
bottom_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=10)
right_frame = tk.Frame(root, bg="pale turquoise")
right_frame.grid(row=0, column=1, rowspan=3, sticky="nsew", padx=20, pady=10)

root.grid_columnconfigure(0, weight=3)
root.grid_columnconfigure(1, weight=2)
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=2)
root.grid_rowconfigure(2, weight=1)

# Top frame
maintitle = tk.Label(top_frame, text="Number Predictor Thing", font=("Helvetica", 40, "bold"), bg="white")
maintitle.pack(pady=10)
entry = tk.Entry(top_frame, font=("Helvetica", 30))
entry.pack(pady=10)
entry.bind("<Return>", numinput)   # safe tkinter binding — replaces keyboard module

# Middle frame
img_button = tk.PhotoImage(file=checknumberbutton)
img_907button = tk.PhotoImage(file=standardizedtestbutton)
check_button = tk.Button(middle_frame, image=img_button, borderwidth=0, compound=tk.CENTER, bg="pale turquoise")
check_button.grid(row=0, column=0, padx=20, pady=20)
button907 = tk.Button(middle_frame, image=img_907button, borderwidth=0, compound=tk.CENTER, bg="pale turquoise")
button907.grid(row=0, column=1, padx=20, pady=20)

# Bottom frame
result_label = tk.Label(bottom_frame, text="            ", font=("Helvetica", 50), bg="skyblue1")
result_label.pack(pady=10)
winorloselabel = tk.Label(bottom_frame, text="", font=("Helvetica", 50), bg="pale turquoise")
winorloselabel.pack(pady=10)
botplayedlabel = tk.Label(bottom_frame, text="AI Win Rate: NA%\nRounds Played: 0", font=('Helvetica', 30, 'bold'), fg='black', bg="pale turquoise")
botplayedlabel.pack(pady=10)

# Right frame
confidenceinit = {str(i).zfill(2): 0 for i in range(0, 101)}
confidence_str = ""
for key, value in confidenceinit.items():
    confidence_str += f"{key}: {value:.2f}, "
    if int(key) % 6 == 0:
        confidence_str += "\n"
confidencelabel = tk.Label(
    right_frame,
    text=f"Confidence levels for prior number:\n{confidence_str}\n(don't use these to cheat weirdo)",
    fg='black', bg="pale turquoise", font=('Helvetica', 15, 'bold')
)
confidencelabel.pack(pady=20)

check_button.bind("<Button-1>", numinput)
button907.bind("<Button-1>", autonuminput)

if __name__ == "__main__":
    root.mainloop()