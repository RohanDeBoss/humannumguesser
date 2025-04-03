import math
import keyboard
import pygame
import threading
import tkinter as tk
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from data import dataset, firstdataset, seconddataset, testsample, frequency, frequency2
import warnings
from functools import lru_cache

warnings.filterwarnings("ignore")

# File paths (update these as needed)
assets = r"humannumguesser/assets/"
checknumberbutton = assets + r"images/check.png"
standardizedtestbutton = assets + r"images/run907.png"
correctsfx = assets + r"audios/correct.mp3"
wrongsfx = assets + r"audios/wrong.mp3"

# Initialize pygame mixer and pre-load sound effects
pygame.mixer.init()
pygame.mixer.music.set_volume(0.25)
correct_sound = pygame.mixer.Sound(correctsfx)
wrong_sound = pygame.mixer.Sound(wrongsfx)

# Global variables
inputted, firstinp, secondinp, temp, tempc, win, played = [], [], [], [], [], 0, []
confidence_base = {str(i).zfill(2): 0 for i in range(0, 101)}
confidence = confidence_base.copy()  # Initialize confidence globally

# --- Machine Learning / Prediction Functions ---

def prepare_data(sequence, n_lags=2):
    """Prepare data using NumPy's sliding window view for efficiency."""
    seq = np.array(sequence)
    n = len(seq) - n_lags
    if n <= 0:
        raise ValueError("short")
    X = np.lib.stride_tricks.sliding_window_view(seq, n_lags + 1)[:, :-1]
    y = np.lib.stride_tricks.sliding_window_view(seq, n_lags + 1)[:, -1]
    return X, y

def predict_next(sequence, n_lags=2):
    """Predict the next number using RandomForestRegressor with parallel processing."""
    if len(sequence) < n_lags + 1:
        raise ValueError("short")
    X, y = prepare_data(sequence, n_lags)
    if X.size == 0 or y.size == 0:
        raise ValueError("short")
    model = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1, warm_start=True)
    model.fit(X, y)
    last_values = np.array(sequence[-n_lags:]).reshape(1, -1)
    return model.predict(last_values)[0]

@lru_cache(maxsize=128)
def normal_pdf(x, mean, sigma):
    """Cached normal probability density function for performance."""
    factor = 1 / (sigma * (2 * math.pi)**0.5)
    exponent = -((x - mean)**2) / (2 * sigma**2)
    return factor * math.exp(exponent)

def normaldist(target_first_digit, target_second_digit, weight):
    """Update confidence based on normal distribution."""
    global confidence
    for key in confidence:
        first_digit = 10 if key == "100" else int(key[0])
        distance = abs(first_digit - target_first_digit)
        confidence[key] += normal_pdf(distance, 0, 2) * weight
        second_digit = int(key[1])
        distance = abs(second_digit - target_second_digit)
        confidence[key] += (2 - (distance / 10)) * weight
        if int(key) == (target_first_digit * 10 + target_second_digit):
            confidence[key] += 1 * weight
        if int(key[1]) == target_second_digit:
            confidence[key] += 0.5 * weight

def othernormaldist(target_number, weight):
    """Update confidence for whole number predictions."""
    global confidence
    for key in confidence:
        number = int(key)
        distance = abs(number - target_number)
        confidence[key] += 10 * normal_pdf(distance, 0, 10) * weight
        if number == target_number:
            confidence[key] += 0.35 * weight

def build_markov_chain(data, k):
    """Build a Markov chain for sequence prediction."""
    markov_chain = {}
    for i in range(len(data) - k):
        current_state = tuple(data[i:i+k])
        next_state = data[i + k]
        markov_chain.setdefault(current_state, {}).setdefault(next_state, 0)
        markov_chain[current_state][next_state] += 1
    return markov_chain

def predict_next_elementmark(markov_chain, current_state):
    """Predict next element using Markov chain."""
    while current_state not in markov_chain and len(current_state) > 1:
        current_state = current_state[1:]
    if current_state in markov_chain:
        transitions = markov_chain[current_state]
        total = sum(transitions.values())
        if total > 0:
            return max(transitions, key=lambda k: transitions[k] / total)
    overall_transitions = {}
    for state, trans in markov_chain.items():
        for next_state, count in trans.items():
            overall_transitions[next_state] = overall_transitions.get(next_state, 0) + count
    if overall_transitions:
        total = sum(overall_transitions.values())
        if total > 0:
            return max(overall_transitions, key=lambda k: overall_transitions[k] / total)
    return None

def try_predict(train_data, method='rf'):
    """Helper function to streamline prediction attempts."""
    try:
        if method == 'rf':
            return round(float(predict_next(train_data)))
        elif method == 'markov':
            markov_chain = build_markov_chain(train_data, 1)
            return int(predict_next_elementmark(markov_chain, tuple(train_data[-1:])))
    except:
        return None

def differencepred():
    """Predict next digits using multiple methods."""
    global confidence, firstinp, secondinp, inputted
    confidence = confidence_base.copy()
    if not inputted:
        return confidence
    try:
        last = inputted[-1]
        firstinp.append(10 if last == "100" else int(last[0]))
        secondinp.append(int(last[1]))
    except:
        pass

    # Random Forest predictions
    nextfirstdiff = try_predict(firstdataset + firstinp)
    nextseconddiff = 0 if nextfirstdiff == 10 else try_predict(seconddataset + secondinp)
    if nextfirstdiff and nextseconddiff is not None:
        normaldist(nextfirstdiff, nextseconddiff, 1)

    # Frequency-based predictions
    try:
        nextfirstdiff, nextseconddiff = frequency[inputted[-1]]
        nextseconddiff = 0 if nextfirstdiff == 10 else nextseconddiff
        normaldist(nextfirstdiff, nextseconddiff, 1.1)
    except:
        pass

    # Markov chain predictions
    nextfirstdiff = try_predict(firstdataset + firstinp, 'markov')
    nextseconddiff = 0 if nextfirstdiff == 10 else try_predict(seconddataset + secondinp, 'markov')
    if nextfirstdiff and nextseconddiff is not None:
        normaldist(nextfirstdiff, nextseconddiff, 1.7)

    # XGBoost predictions
    try:
        window_size = 10
        if len(firstinp) > window_size:
            X_train, y_train = [], []
            for i in range(len(firstinp) - window_size):
                group = firstinp[i:i+window_size]
                stats = [np.mean(group), np.std(group), np.median(group), np.max(group), np.min(group), np.max(group) - np.min(group)]
                X_train.append(stats)
                y_train.append(firstinp[i+window_size])
            model = xgb.XGBRegressor(n_estimators=35, max_depth=10, learning_rate=0.11, objective='reg:squarederror', nthread=-2, tree_method='hist')
            model.fit(np.array(X_train), np.array(y_train))
            next_group = firstinp[-window_size:]
            stats = [np.mean(next_group), np.std(next_group), np.median(next_group), np.max(next_group), np.min(next_group), np.max(next_group) - np.min(next_group)]
            nextfirstdiff = int(model.predict(np.array([stats]))[0])
            nextseconddiff = 0 if nextfirstdiff == 100 else None
            if nextfirstdiff != 100 and len(secondinp) > window_size:
                X_train, y_train = [], []
                for i in range(len(secondinp) - window_size):
                    group = secondinp[i:i+window_size]
                    stats = [np.mean(group), np.std(group), np.median(group), np.max(group), np.min(group), np.max(group) - np.min(group)]
                    X_train.append(stats)
                    y_train.append(secondinp[i+window_size])
                model.fit(np.array(X_train), np.array(y_train))
                next_group = secondinp[-window_size:]
                stats = [np.mean(next_group), np.std(next_group), np.median(next_group), np.max(next_group), np.min(next_group), np.max(next_group) - np.min(next_group)]
                nextseconddiff = int(model.predict(np.array([stats]))[0])
            if nextseconddiff is not None:
                normaldist(nextfirstdiff, nextseconddiff, 1.1)
    except:
        pass

    # Whole number predictions
    train = dataset + inputted
    for method, weight in [('rf', 8), ('markov', 4.6)]:
        next_num = try_predict(train, method)
        if next_num:
            othernormaldist(next_num, weight)
    try:
        othernormaldist(frequency2[inputted[-1]], 4.8)
    except:
        pass
    return confidence

def main():
    """Main prediction logic combining multiple strategies."""
    global inputted, temp, tempc, confidence, firstinp, secondinp
    confidence = differencepred()
    for i, val in enumerate(dataset):
        confidence[val] += (20609 + len(inputted)) / 7500000
        try:
            for j in range(2, min(1000002, len(dataset) - i)):
                temp, tempc = dataset[max(0, i-j):i][::-1], inputted[-j:][::-1]
                if temp == tempc:
                    confidence[dataset[i + 1]] += (j - 1) * 4.6
                else:
                    break
        except:
            pass

    for i, val in enumerate(inputted):
        retro = i / len(inputted)
        confidence[val] += 0.7 * retro
        try:
            for j in range(2, min(1000002, len(inputted) - i)):
                temp, tempc = inputted[max(0, i-j):i][::-1], inputted[-j:][::-1]
                if temp == tempc:
                    confidence[inputted[i + 1]] += (j - 1) * 10.9 * retro
                else:
                    break
        except:
            pass

    # Pattern-based adjustments
    if len(inputted) >= 2:
        diff = int(inputted[-2]) - int(inputted[-1])
        if diff in {1, 2, 3, 5, 10, 20, -1, -2, -3, -5, -10, -20}:
            next_element = int(inputted[-1]) + diff
            if 0 <= next_element <= 100:
                confidence[str(next_element).zfill(2)] += 10
        if len(inputted) >= 3 and inputted[-1] != inputted[-2] and diff == int(inputted[-2]) - int(inputted[-3]):
            next_element = int(inputted[-1]) + diff
            if 0 <= next_element <= 100:
                confidence[str(next_element).zfill(2)] += 30
        try:
            ratio = int(inputted[-2]) / int(inputted[-1])
            if ratio in {2, 0.5}:
                next_element = int(int(inputted[-1]) * ratio)
                if 0 <= next_element <= 100:
                    confidence[str(next_element).zfill(2)] += 7
            if len(inputted) >= 4:
                ratios = [int(inputted[i]) / int(inputted[i-1]) for i in range(-3, 0)]
                if all(r == ratios[0] for r in ratios):
                    next_element = int(int(inputted[-1]) * ratios[0])
                    if 0 <= next_element <= 100:
                        confidence[str(next_element).zfill(2)] += 30
        except:
            pass

    if not inputted:
        return "37"
    try:
        if len(played) >= 2 and inputted[-1] == played[1] and inputted[-2] == played[2]:
            return played[0]
    except:
        pass
    max_conf = max(confidence.values())
    return inputted[-1] if max_conf == 0 else max(confidence, key=confidence.get)

# --- Tkinter UI Functions ---

def start_ai_thread(input_text):
    """Run AI prediction in a separate thread."""
    def run():
        returned = main()
        root.after(0, update_ui_after_ai, input_text, returned)
    threading.Thread(target=run, daemon=True).start()

def update_confidence_display():
    """Batch confidence string updates for efficiency."""
    confidence_str = []
    for i, (key, value) in enumerate(confidence.items()):
        confidence_str.append(f"{key}: {value:.1f}")
        if (i + 1) % 6 == 0:
            confidence_str.append("\n")
    return "".join(confidence_str)

def update_ui_after_ai(input_text, returned):
    """Update UI after AI prediction."""
    global win
    cleanup()
    inputted.append(str(input_text).zfill(2) if 0 <= int(input_text) <= 9 else input_text)
    played.insert(0, returned)
    if len(played) >= 4:
        played.pop(-1)
    if inputted[-1] == returned:
        correct_sound.play()
        result_label.config(text=f"    {returned}    ", bg="lawn green")
        win += 1
        winorloselabel.config(text="Bot Wins")
    else:
        wrong_sound.play()
        result_label.config(text=f"    {returned}    ", bg="red2")
        winorloselabel.config(text="Bot Lost")
    botplayedlabel.config(text=f"AI Win Rate: {(win/len(inputted)*100):.3f}%\nRounds Played: {len(inputted)}")
    confidencelabel.config(text=f"Confidence levels:\n{update_confidence_display()}\n", fg='black', bg="pale turquoise")
    result_label.after(200, lambda: [result_label.config(bg="skyblue1"), entry.focus_set()])

def numinput(event):
    """Handle user input."""
    try:
        if timerup or len(inputted) >= 400:
            raise ValueError
        input_text = entry.get()
        entry.delete(0, "end")
        result_label.config(text="            ")
        if 0 <= int(input_text) <= 100 and (input_text[0] != "0" or input_text == "0"):
            start_ai_thread(input_text)
        else:
            raise ValueError
    except ValueError:
        result_label.config(text="poopy number", bg="skyblue1")
    entry.focus_set()

def start_autotest_thread():
    """Run autotest in a separate thread."""
    def run():
        global win
        for input_text in testsample:
            returned = main()
            inputted.append(input_text)
            if input_text == returned:
                win += 1
            print(f"actual: {input_text} AI winrate {(win/len(inputted)*100):.3f}% Rounds {len(inputted)}/907")
        root.after(0, update_ui_after_autotest)
    threading.Thread(target=run, daemon=True).start()

def update_ui_after_autotest():
    """Update UI after autotest."""
    result_label.config(text=" Done ")
    botplayedlabel.config(text=f"AI Win Rate: {(win/len(inputted)*100):.3f}%\nRounds Played: {len(inputted)}")
    confidencelabel.config(text=f"Confidence levels:\n{update_confidence_display()}\n(don't cheat!)", fg='black', bg="pale turquoise")
    entry.focus_set()

def autonuminput(event):
    """Trigger autotest."""
    result_label.config(text="calculating")
    start_autotest_thread()

def cleanup():
    """Manage memory by trimming lists."""
    global inputted, firstinp, secondinp, temp, tempc
    if len(inputted) > 1000:
        inputted = inputted[-500:]
        firstinp = firstinp[-500:]
        secondinp = secondinp[-500:]
        temp.clear()
        tempc.clear()

# --- UI Initialization ---

keyboard.on_press_key("enter", numinput)
timerup = False

root = tk.Tk()
root.title("Number Predictor Thing")
root.configure(bg="pale turquoise")
root.attributes("-fullscreen", True)

root.bind("<Escape>", lambda e: [root.attributes("-fullscreen", False), entry.focus_set()])

# Layout frames
for i, frame in enumerate([tk.Frame(root, bg="pale turquoise") for _ in range(4)]):
    frame.grid(row=i//2*2, column=i%2, sticky="nsew", padx=20, pady=10 if i < 2 else (10, 20))
top_frame, middle_frame, bottom_frame, right_frame = root.winfo_children()

root.grid_columnconfigure(0, weight=3)
root.grid_columnconfigure(1, weight=2)
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=2)

# Top Frame
tk.Label(top_frame, text="Number Predictor Thing", font=("Helvetica", 40, "bold"), bg="white").pack(pady=10)
entry = tk.Entry(top_frame, font=("Helvetica", 30))
entry.pack(pady=10)
entry.focus_set()

# Middle Frame
img_button = tk.PhotoImage(file=checknumberbutton)
img_907button = tk.PhotoImage(file=standardizedtestbutton)
check_button = tk.Button(middle_frame, image=img_button, borderwidth=0, bg="pale turquoise")
check_button.grid(row=0, column=0, padx=20, pady=20)
button907 = tk.Button(middle_frame, image=img_907button, borderwidth=0, bg="pale turquoise")
button907.grid(row=0, column=1, padx=20, pady=20)

# Bottom Frame
custom_frame = tk.Frame(bottom_frame, bg="pale turquoise")
custom_frame.grid(row=0, column=0, sticky="nsew")
result_label = tk.Label(custom_frame, text="            ", font=("Helvetica", 50), bg="skyblue1")
result_label.grid(row=0, column=0, pady=10)
winorloselabel = tk.Label(custom_frame, text="", font=("Helvetica", 50), bg="pale turquoise")
winorloselabel.grid(row=1, column=0, pady=10)
botplayedlabel = tk.Label(custom_frame, text="AI Win Rate: NA%\nRounds Played: 0", font=('Helvetica', 30, 'bold'), fg='black', bg="pale turquoise")
botplayedlabel.grid(row=2, column=0, pady=10)
bottom_frame.grid_columnconfigure(0, weight=1)

# Right Frame
confidencelabel = tk.Label(right_frame, text=f"Confidence levels:\n{update_confidence_display()}\n(don't cheat!)", fg='black', bg="pale turquoise", font=('Helvetica', 15, 'bold'))
confidencelabel.pack(pady=20)

# Bind buttons
for btn, func in [(check_button, numinput), (button907, autonuminput)]:
    btn.bind("<Button-1>", func)
    btn.bind("<ButtonRelease-1>", lambda e: entry.focus_set())

if __name__ == "__main__":
    root.mainloop()