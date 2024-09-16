'''
This file contains code to preprocess data into annotation files.
Input the starting directory, ignore folders and data type to start evaluating.
splitRatio will assign the indicated ratio amount to the first in the list of annotationFiles.
'''
import random
import json
import os
from sqlite3 import SQLITE_SAVEPOINT
from speechbrain.dataio.dataio import read_audio # Because this file is used by fast audio, which has speechbrain, this lib will be used.

# The files needed for 

startDir = "C:/Users/jazzt/Documents/GitHub/Deepfake-Audio-Detection/data/SG"
ignoreFolders = ["test", "smallTest"]
annotationFile = "eval"+"_annotation.json"
codeDir = "./data"

splitRatio = 0 # if 0, no split. Must be between 0 and 1
annotationFiles = ["train_annotation.json","dev_annotation.json"]

# Loop through all the files in start Dir. 
# If it's a dir and it's in ignoreFolders, skip it.
# If it's a file, add it to the annotationFile

annotations = {}
SAMPLERATE = 16000
SEED = 1234

def main():
    
    # Populate annotations dict
    FileLooper(startDir, AddAnnotation)
    spoofCount, bonafideCount = CountSpoofnBonafide(annotations)
    print(f"FileLooper finish, total files in annotations: {len(annotations)}, spoof: {spoofCount}, bonafide: {bonafideCount}")

    if splitRatio > 0:
        print("Splitting")
        dict1, dict2 = SplitDict(annotations)

        spoofCount, bonafideCount = CountSpoofnBonafide(dict1)
        print(f"Dict1 total: {len(dict1)}, spoof: {spoofCount}, bonafide: {bonafideCount}")
        spoofCount, bonafideCount = CountSpoofnBonafide(dict2)
        print(f"Dict2 total: {len(dict2)}, spoof: {spoofCount}, bonafide: {bonafideCount}")

        SaveToJson(dict1, annotationFiles[0])
        SaveToJson(dict2, annotationFiles[1])
    else:
        SaveToJson(annotations, annotationFile)

    return 0

def FileLooper(dir_path, fptr):
    for item in os.listdir(dir_path):
        item_path = f"{dir_path}/{item}"
        if os.path.isdir(item_path) and (os.path.basename(item_path) not in ignoreFolders):
            print(f"Entering folder: {item_path}")
            FileLooper(item_path, fptr)
        else:
            fptr(item_path)

def AddAnnotation(item_path: str):
    if ".wav" not in item_path:
        print(f"Ignoring {item_path}, not a wav file")
        return
    signal = read_audio(item_path)
    duration = signal.shape[0] / SAMPLERATE
    id = os.path.splitext(os.path.basename(item_path))[0]
    spoofType = "spoof" if len(id) > 12 else "bonafide"
    _path = item_path.replace(startDir, codeDir)
    
    annotations[id] = {
        'file_path': _path,
        'duration': duration,
        'key': spoofType
    }

def SplitDict(ogDict: dict, ratio = 0.9):
    print("SplitDict")
    items = list(ogDict.items())

    random.seed(SEED)
    random.shuffle(items)

    split_index = int(len(items) * ratio)

    first_part = items[:split_index]
    second_part = items[split_index:]

    dict1 = dict(first_part) # dict1 will contain the amount indicated by ratio
    dict2 = dict(second_part)

    return dict1, dict2

def SaveToJson(item: dict, jsonFilePath: str):
    ''' Save the dictionary to a json file. '''
    print("Saving to json")
    with open(jsonFilePath, 'w') as f:
        json.dump(item, f)
        print(f"Features are saved to {jsonFilePath}")

def CountSpoofnBonafide(d: dict):
    spoofCount = 0
    for key in d:
        if d[key]["key"] == "spoof":
            spoofCount += 1
    
    return spoofCount, len(d) - spoofCount

main()









