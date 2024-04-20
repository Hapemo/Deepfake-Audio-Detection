# This script is used to prep the data required for training in a smaller and more controlled quantity of data.

import math
import random
import shutil

# shutil.copy(src + file + filetype, dst + file + filetype)

srcPath = "C:/Users/jazzt/Documents/GitHub/Deepfake-Audio-Detection/data/LA/LA/"
dstPath = "C:/Users/jazzt/Documents/GitHub/Deepfake-Audio-Detection/data/LA/subset_LA/"
cmMetaDataPath = "ASVspoof2019_LA_cm_protocols/"


attack_types = [f'A{_id:02d}' for _id in range(1, 20)]

MasterContainer = { attack_type: [] for attack_type in attack_types }
MasterContainer['-'] = []

catList = {
    "dev": ("ASVspoof2019_LA_dev/flac/", "ASVspoof2019.LA.cm.dev.trl.txt"),
    "eval": ("ASVspoof2019_LA_eval/flac/", "ASVspoof2019.LA.cm.eval.trl.txt"),
    "train": ("ASVspoof2019_LA_train/flac/", "ASVspoof2019.LA.cm.train.trn.txt")
}

# Load in the meta file
# Read how many data from each category are there
# Identify 10% of the data and append that to the new dataset

target = catList["eval"]

inFile = open(srcPath+cmMetaDataPath+target[1], "r")
outFile = open(dstPath+cmMetaDataPath+target[1], "w")

for line in inFile.readlines():
    MasterContainer[line.split(' ')[3]].append(line)


for key in MasterContainer:
    # print(f'{key}: {len(MasterContainer[key])}') # print count of data in each category
    
    # Pick out 10% random files
    nb_of_files = math.floor(len(MasterContainer[key]) / 10)
    random.shuffle(MasterContainer[key])
    MasterContainer[key] = MasterContainer[key][:nb_of_files]
    
    for item in MasterContainer[key]:
        shutil.copy(srcPath + target[0] + item.split(' ')[1] + ".flac", dstPath + target[0] + item.split(' ')[1] + ".flac")
        outFile.write(item)


# print(MasterContainer)






