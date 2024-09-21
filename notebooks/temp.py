import os
import shutil
import random
from numpy import character

startDir = "C:/Users/jazzt/Documents/GitHub/Deepfake-Audio-Detection/data/showcase_samples/bonafide"

for folder in os.listdir(startDir):
    characterFolder = os.path.join(startDir, folder)
    if not os.path.isdir(characterFolder): continue

    for subfolder in os.listdir(characterFolder):
        actualFolder = os.path.join(characterFolder, subfolder)
        if not os.path.isdir(actualFolder): continue

        content = os.listdir(actualFolder)
        if len(content) <= 65: continue

        random.shuffle(content)
        toDelete = content[65:]
        
        for file in toDelete:
            deleteDir = os.path.join(actualFolder, file)
            print(deleteDir)
            # os.remove(deleteDir)











