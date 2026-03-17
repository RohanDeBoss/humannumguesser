# create_filtered_data_v4: 11.797!

""" v4:
Baseline no filter: 11.577
Spam only (4, -136): 11.025
          (3, -247): 10.805
          (2, -819): 11.246
          (5, -92): 11.136
          (6, -67): 10.915
          (7, -55): 11.025
          (8, -48): 11.466
          (9, -32): 11.356
          (10, -23): 11.687
          (23, -23): 11.687

Counting only (20, -331): 11.466
              (19, -369): 11.136
              (15, -419): 10.754
              (10, -558): 10.915
              (25, -249): 10.695
              (45, -54): 11.025
Both: 
              23 & 20: 11.797!
"""

import os
from data import dataset

# Counters for the summary
removed_spam = 0
removed_count = 0
removed_alt = 0

def find_spam_sequence(data, index, min_len=23):
    global removed_spam
    if index > len(data) - min_len: return None
    val = data[index]
    if all(data[index+j] == val for j in range(min_len)):
        end_index = index + min_len
        while end_index < len(data) and data[end_index] == val: end_index += 1
        removed_spam += (end_index - index)
        print(f"  -> [Spam] Removed '{val}' repeating {end_index - index} times at index {index}.")
        return end_index
    return None

def find_counting_sequence(data, index, min_len=20):
    global removed_count
    if index > len(data) - min_len: return None
    try:
        if all(int(data[index+j]) == int(data[index]) + j for j in range(min_len)):
            end_index = index + min_len
            while end_index < len(data) and int(data[end_index]) == int(data[index]) + (end_index - index): end_index += 1
            removed_count += (end_index - index)
            print(f"  -> [Counting Up] Removed sequence of {end_index - index} starting at {data[index]}.")
            return end_index
    except: pass
    try:
        if all(int(data[index+j]) == int(data[index]) - j for j in range(min_len)):
            end_index = index + min_len
            while end_index < len(data) and int(data[end_index]) == int(data[index]) - (end_index - index): end_index += 1
            removed_count += (end_index - index)
            print(f"  -> [Counting Down] Removed sequence of {end_index - index} starting at {data[index]}.")
            return end_index
    except: pass
    return None
    
def find_alternating_sequence(data, index, min_len=8):
    global removed_alt
    if index > len(data) - min_len: return None
    val_a, val_b = data[index], data[index+1]
    if val_a == val_b: return None
    if all(data[index+j] == (val_a if j % 2 == 0 else val_b) for j in range(min_len)):
        end_index = index + min_len
        while end_index < len(data) and data[end_index] == (val_a if (end_index - index) % 2 == 0 else val_b): end_index += 1
        removed_alt += (end_index - index)
        print(f"  -> [Alternating] Removed '{val_a}' and '{val_b}' repeating {end_index - index} times.")
        return end_index
    return None

def filter_all_low_quality(original_data):
    filtered_data = []
    i, n = 0, len(original_data)
    while i < n:
        end_index = find_spam_sequence(original_data, i) or find_counting_sequence(original_data, i) or find_alternating_sequence(original_data, i)
        if end_index: i = end_index
        else:
            filtered_data.append(original_data[i])
            i += 1
    return filtered_data

if __name__ == "__main__":
    print("Starting data cleaning process...\n")
    dataset_filtered = filter_all_low_quality(dataset)
    
    firstdataset_filtered = []
    seconddataset_filtered = []
    
    for number_str in dataset_filtered:
        if number_str == "100":
            firstdataset_filtered.append('10')
            seconddataset_filtered.append('0')
        else:
            # Strictly keeping strings to protect Markov Chain indexing
            firstdataset_filtered.append(number_str[0])
            seconddataset_filtered.append(number_str[1])
            
    _dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(_dir, "data_filtered_v4.py")
    
    with open(output_filename, "w") as f:
        f.write(f"dataset_filtered = {dataset_filtered}\n\n")
        f.write(f"firstdataset_filtered = {firstdataset_filtered}\n\n")
        f.write(f"seconddataset_filtered = {seconddataset_filtered}\n")
        
    print("\n--- Filtering Summary ---")
    print(f"Original dataset length: {len(dataset)}")
    print(f"Filtered dataset length: {len(dataset_filtered)}")
    print(f"Total entries removed:   {len(dataset) - len(dataset_filtered)}")
    print(f"  - Spam removed:        {removed_spam}")
    print(f"  - Counting removed:    {removed_count}")
    print(f"  - Alternating removed: {removed_alt}")
    print("-------------------------\n")
    print(f"Done! Data written to '{output_filename}'.")