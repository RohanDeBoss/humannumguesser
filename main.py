import math, keyboard, pygame
import tkinter as tk
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from data import dataset, firstdataset, seconddataset, testsample, frequency, frequency2
import warnings
warnings.filterwarnings("ignore")

# Change these variables to the path of the files
assets = r"humannumguesser/assets/"
checknumberbutton = assets + r"images/check.png"
standardizedtestbutton = assets + r"images/run907.png"
correctsfx = assets + r"audios/correct.mp3"
wrongsfx = assets + r"audios/wrong.mp3"

# Global variables for the prediction functions
global temp, tempc, next_element, confidence, nextfirstdiff, nextseconddiff
inputted, firstdiff, seconddiff, temp, tempc, win, train, firstinp, secondinp, played = [], [], [], [], [], 0, [], [], [], []

# ----- Machine Learning / Prediction Functions -----
def prepare_data(sequence, n_lags=2):
    X, y = [], []
    for i in range(len(sequence) - n_lags):
        X.append(sequence[i:i + n_lags])
        y.append(sequence[i + n_lags])
    return np.array(X), np.array(y)

def predict_next(sequence, n_lags=2):
    if len(sequence) < n_lags + 1: 
        raise ValueError("short")
    X, y = prepare_data(sequence, n_lags)
    if X.size == 0 or y.size == 0: 
        raise ValueError("short")
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    last_values = np.array(sequence[-n_lags:]).reshape(1, -1)
    next_number = model.predict(last_values)
    return next_number

def normal_pdf(x, mean, sigma):
    factor = 1 / (sigma * (2 * 3.141592653589793)**0.5)
    exponent = -((x - mean)**2) / (2 * sigma**2)
    return factor * (2.718281828459045**exponent)

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
        current_state = tuple(data[i:i+k])
        next_state = data[i + k]
        if current_state not in markov_chain:
            markov_chain[current_state] = {}
        if next_state not in markov_chain[current_state]:
            markov_chain[current_state][next_state] = 0
        markov_chain[current_state][next_state] += 1
    return markov_chain

def predict_next_elementmark(markov_chain, current_state):
    while current_state not in markov_chain and len(current_state) > 1:
        current_state = current_state[1:]
    if current_state in markov_chain:
        transitions = markov_chain[current_state]
        total_count = sum(transitions.values())
        if total_count > 0:
            probabilities = {state: count / total_count for state, count in transitions.items()}
            next_state = max(probabilities, key=probabilities.get)
            return next_state
    overall_transitions = {}
    for state, transitions in markov_chain.items():
        for next_state, count in transitions.items():
            overall_transitions[next_state] = overall_transitions.get(next_state, 0) + count
    if overall_transitions:
        total_count = sum(overall_transitions.values())
        if total_count > 0:
            probabilities = {state: count / total_count for state, count in overall_transitions.items()}
            next_state = max(probabilities, key=probabilities.get)
            return next_state
    return None

def differencepred():
    global nextfirstdiff, nextseconddiff, confidence, firstinp, secondinp, inputted
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
    nextfirstdiff, nextseconddiff = None, None
    train = firstdataset + firstinp
    try: 
        nextfirstdiff = round(float(predict_next(train)))
    except ValueError: 
        pass
    if nextfirstdiff == 10: 
        nextseconddiff = 0
    else:
        train = seconddataset + secondinp
        try: 
            nextseconddiff = round(float(predict_next(train)))
        except ValueError: 
            pass
    if nextseconddiff and nextfirstdiff: 
        normaldist(nextfirstdiff, nextseconddiff, 1)
    nextfirstdiff, nextseconddiff = None, None
    try:
        nextfirstdiff = frequency[inputted[-1]][0]
        if nextfirstdiff == 10: 
            nextseconddiff = 0
        else: 
            nextseconddiff = frequency[inputted[-1]][1]
        normaldist(nextfirstdiff, nextseconddiff, 1.1)
    except: 
        pass
    nextfirstdiff, nextseconddiff = None, None
    train = firstdataset + firstinp
    try:
        markov_chain = build_markov_chain(train, 1)
        current_state = tuple(train[-1:])
        nextfirstdiff = int(predict_next_elementmark(markov_chain, current_state))
    except: 
        pass
    if nextfirstdiff == 10: 
        nextseconddiff = 0
    else:
        try:
            train = seconddataset + secondinp
            markov_chain = build_markov_chain(train, 1)
            current_state = tuple(train[-1:])
            nextseconddiff = int(predict_next_elementmark(markov_chain, current_state))
        except: 
            pass
    if nextseconddiff and nextfirstdiff: 
        normaldist(nextfirstdiff, nextseconddiff, 1.7)
    nextfirstdiff, nextseconddiff = None, None
    try:
        X_train = []
        y_train = []
        window_size = 10
        for i in range(len(firstinp) - window_size):
            group = firstinp[i:i+window_size]
            mean = np.mean(group)
            std_dev = np.std(group)
            median = np.median(group)
            max_val = np.max(group)
            min_val = np.min(group)
            range_val = max_val - min_val
            X_train.append([mean, std_dev, median, max_val, min_val, range_val])
            y_train.append(firstinp[i+window_size])
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        model = xgb.XGBRegressor(n_estimators=35, max_depth=10, learning_rate=0.11, objective='reg:squarederror')
        model.fit(X_train, y_train)
        next_group = firstinp[-window_size:]
        mean = np.mean(next_group)
        std_dev = np.std(next_group)
        median = np.median(next_group)
        max_val = np.max(next_group)
        min_val = np.min(next_group)
        range_val = max_val - min_val
        nextfirstdiff = int(model.predict(np.array([[mean, std_dev, median, max_val, min_val, range_val]])))
    except: 
        pass
    if nextfirstdiff == 100: 
        nextseconddiff = 0
    else:
        try:
            X_train = []
            y_train = []
            window_size = 10
            for i in range(len(secondinp) - window_size):
                group = secondinp[i:i+window_size]
                mean = np.mean(group)
                std_dev = np.std(group)
                median = np.median(group)
                max_val = np.max(group)
                min_val = np.min(group)
                range_val = max_val - min_val
                X_train.append([mean, std_dev, median, max_val, min_val, range_val])
                y_train.append(secondinp[i+window_size])
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            model = xgb.XGBRegressor(n_estimators=35, max_depth=10, learning_rate=0.11, objective='reg:squarederror')
            model.fit(X_train, y_train)
            next_group = secondinp[-window_size:]
            mean = np.mean(next_group)
            std_dev = np.std(next_group)
            median = np.median(next_group)
            max_val = np.max(next_group)
            min_val = np.min(next_group)
            range_val = max_val - min_val
            nextseconddiff = int(model.predict(np.array([[mean, std_dev, median, max_val, min_val, range_val]])))
        except: 
            pass
    if nextseconddiff and nextfirstdiff: 
        normaldist(nextfirstdiff, nextseconddiff, 1.1)
    train = dataset + inputted
    nextfirstdiff = None
    try: 
        nextfirstdiff = round(float(predict_next(train)))
    except: 
        pass
    if nextseconddiff: 
        othernormaldist(int(nextfirstdiff), 8)
    nextfirstdiff = None
    try:
        markov_chain = build_markov_chain(train, 1)
        current_state = tuple(train[-1:])
        nextfirstdiff = int(predict_next_elementmark(markov_chain, current_state))
    except: 
        pass
    if nextfirstdiff: 
        othernormaldist(int(nextfirstdiff), 4.6)
    nextfirstdiff = None
    try: 
        nextfirstdiff = frequency2[inputted[-1]]
    except: 
        pass
    if nextfirstdiff: 
        othernormaldist(int(nextfirstdiff), 4.8)
    return confidence

def main():
    global inputted, retro, temp, tempc, next_element, confidence, firstinp, secondinp
    next_element, difference = 0, 0
    confidence = differencepred()
    for i in range(len(dataset)):
        confidence[dataset[i]] += (20609 + len(inputted)) / 7500000
        try:
            for j in range(2, min(1000002, len(dataset) - i)):
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
    for i in range(len(inputted)):
        retro = i / (len(inputted))
        confidence[inputted[i]] += 0.7 * retro
        for j in range(2, min(1000002, len(inputted) - i)):
            temp, tempc = [], []
            for k in range(j):
                temp.insert(0, inputted[i - k])
                tempc.insert(0, inputted[-1 - k])
            if temp == tempc: 
                confidence[inputted[i + 1]] += (j - 1) * 10.9 * retro
            else: 
                break
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
        ratios = [int(inputted[i]) / int(inputted[i-1]) for i in range(len(inputted) - 3, len(inputted))]
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
    if max(confidence.items()) == 0.0: 
        return inputted[-1]
    inverted_confidence = {v: k for k, v in confidence.items()}
    return inverted_confidence[max(confidence.values())]

# ----- Tkinter UI Functions -----
def numinput(event):
    global win, confidence, confidencelabel, played, timerup, inputted
    try:
        if (timerup == False) and (len(inputted) < 500):
            input_text = entry.get()
        else:
            print(inputted)
            raise ValueError
        entry.delete(0, "end")
        result_label.config(text="            ")
        if (0 <= int(input_text) <= 100) and ((((input_text[0] not in {"0", " "}) == (0 <= int(input_text)) <= 100)) or input_text == "0"):
            returned = main()
            inputted.append(input_text)
            if 0 <= int(inputted[-1]) <= 9: 
                inputted[-1] = f"0{inputted[-1]}"
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
            botplayedlabel.config(text=f"AI Win Rate: {(win/len(inputted)*100):.3f}%\nRounds Played: {len(inputted)}")
            confidence_str = ""
            result_label.after(200, lambda: result_label.config(bg="skyblue1"))
            for key, value in confidence.items():
                confidence_str += f"{key}: {value:.2f}, "
                if int(key) % 6 == 0:
                    confidence_str += "\n"
            confidencelabel.config(text=f"Confidence levels for prior number:\n{confidence_str}\n(don't use these to cheat weirdo)", fg='black', bg="pale turquoise")
        else:
            raise ValueError
    except ValueError:
        result_label.config(text="poopy number", bg="skyblue1")

def autonuminput(event):
    global win, confidence, confidencelabel, inputted, firstinp, secondinp
    result_label.config(text="calculating")
    for input_text in testsample:
        returned = main()
        inputted.append(input_text)
        if input_text == returned: 
            win += 1
        print(f"actual answer: {input_text} AI winrate {(win/len(inputted)*100):.3f}% Rounds played {len(inputted)}/907")
    botplayedlabel.config(text=f"AI Win Rate: {(win/len(inputted)*100):.3f}%\nRounds Played: {len(inputted)}")
    confidence_str = ""
    for key, value in confidence.items():
        confidence_str += f"{key}: {value:.2f}, "
        if int(key) % 6 == 0:
            confidence_str += "\n"
    result_label.config(text=" Done ")
    confidencelabel.config(text=f"Confidence levels for prior number:\n{confidence_str}\n(don't use these to cheat weirdo)", fg='black', bg="pale turquoise")

# ----- UI Initialization -----
keyboard.on_press_key("enter", numinput)
pygame.mixer.init()
timerup = False

root = tk.Tk()
root.title("Number Predictor Thing")
root.configure(bg="pale turquoise")
root.geometry("1280x620")  # You can keep this for reference
root.attributes("-fullscreen", True)

def toggle_fullscreen(event=None):
    root.attributes("-fullscreen", False)

root.bind("<Escape>", toggle_fullscreen)

# Create layout frames
top_frame = tk.Frame(root, bg="pale turquoise")
top_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=10)
middle_frame = tk.Frame(root, bg="pale turquoise")
middle_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
bottom_frame = tk.Frame(root, bg="pale turquoise")
bottom_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=10)
right_frame = tk.Frame(root, bg="pale turquoise")
right_frame.grid(row=0, column=1, rowspan=3, sticky="nsew", padx=20, pady=10)

# Configure grid weights for responsive scaling
root.grid_columnconfigure(0, weight=3)
root.grid_columnconfigure(1, weight=2)
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=2)
root.grid_rowconfigure(2, weight=1)

# Top Frame: Title and Entry
maintitle = tk.Label(top_frame, text="Number Predictor Thing", font=("Helvetica", 40, "bold"), bg="white")
maintitle.pack(pady=10)
entry = tk.Entry(top_frame, font=("Helvetica", 30))
entry.pack(pady=10)

# Middle Frame: Buttons
img_button = tk.PhotoImage(file=checknumberbutton)
img_907button = tk.PhotoImage(file=standardizedtestbutton)
check_button = tk.Button(middle_frame, image=img_button, borderwidth=0, compound=tk.CENTER, bg="pale turquoise")
check_button.grid(row=0, column=0, padx=20, pady=20)
button907 = tk.Button(middle_frame, image=img_907button, borderwidth=0, compound=tk.CENTER, bg="pale turquoise")
button907.grid(row=0, column=1, padx=20, pady=20)

# Bottom Frame: Results and Win Info
result_label = tk.Label(bottom_frame, text="            ", font=("Helvetica", 50), bg="skyblue1")
result_label.pack(pady=10)
winorloselabel = tk.Label(bottom_frame, text="", font=("Helvetica", 50), bg="pale turquoise")
winorloselabel.pack(pady=10)
botplayedlabel = tk.Label(bottom_frame, text="AI Win Rate: NA%\nRounds Played: 0", font=('Helvetica', 30, 'bold'), fg='black', bg="pale turquoise")
botplayedlabel.pack(pady=10)

# Right Frame: Confidence Levels
confidenceinit = {str(i).zfill(2): 0 for i in range(0, 101)}
confidence_str = ""
for key, value in confidenceinit.items():
    confidence_str += f"{key}: {value:.2f}, "
    if int(key) % 6 == 0:
        confidence_str += "\n"
confidencelabel = tk.Label(right_frame, text=f"Confidence levels for prior number:\n{confidence_str}\n(don't use these to cheat weirdo)", fg='black', bg="pale turquoise", font=('Helvetica', 15, 'bold'))
confidencelabel.pack(pady=20)

# Bind events to buttons
check_button.bind("<Button-1>", numinput)
button907.bind("<Button-1>", autonuminput)

if __name__ == "__main__":
    root.mainloop()
