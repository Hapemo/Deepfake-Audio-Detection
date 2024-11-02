''' This script is responsible for generating the meta files for prompt tuning '''
import os
import shutil

# 0 - Generate meta file
# 1 - Combine meta file
# 2 - Move all nested sound file into one directory
state = 2

if state == 0:
	metaFilePath = "meta1.txt"

	foldersDir = "C:/Users/jazzt/Documents/GitHub/Deepfake-Audio-Detection/data"
	folders = ["SG"]
	blacklistFolders = ["test"]

	nameformat = "- name - - - status - dataset"
	dataset = "progress"

	metafile = open(metaFilePath, "w")

	def isSpoof(name:str):
		return len(name) > 8

	def Loop(folderPath:str):
		for file in os.listdir(folderPath):
			currPath = os.path.join(folderPath, file)
			if os.path.isdir(currPath):
				ignore = False
				for folder in blacklistFolders:
					if file in folder:
						ignore = True
						break
				if ignore: continue
				else:
					Loop(currPath)
			
			# is file
			if ".wav" not in file and ".flac" not in file: continue
			name = file.split(".")[0]
			status = "spoof" if isSpoof(name) else "bonafide"
			info = f"{nameformat.replace('name', name).replace('status', status).replace('dataset', dataset)}\n"
			metafile.write(info)
			
	for folder in folders:
		Loop(os.path.join(foldersDir, folder))
elif state == 1:
	metaFilePaths = ["C:/Users/jazzt/Documents/GitHub/Deepfake-Audio-Detection/meta1.txt", "C:/Users/jazzt/Documents/GitHub/Deepfake-Audio-Detection/meta.txt"]
	combinedMetaFilePath = "meta2.txt"
	
	combineMetaFile = open(combinedMetaFilePath, "w")
	for metafilepath in metaFilePaths:
		metafile = open(metafilepath, "r")
		for line in metafile.readlines(): combineMetaFile.write(line)
elif state == 2:
	finalDir = "C:/Users/jazzt/Documents/GitHub/Deepfake-Audio-Detection/data/testing"
	if not os.path.exists(finalDir): os.makedirs(finalDir)

	foldersDir = "C:/Users/jazzt/Documents/GitHub/Deepfake-Audio-Detection/data"
	folders = ["SG"]
	blacklistFolders = ["smallTest", "test"]

	def Loop(folderPath:str):
		for file in os.listdir(folderPath):
			currPath = os.path.join(folderPath, file)
			if os.path.isdir(currPath):
				ignore = False
				for folder in blacklistFolders:
					if file in folder:
						ignore = True
						break
				if ignore: continue
				else:
					Loop(currPath)
			
			# is file
			if ".wav" not in file and ".flac" not in file: continue
			shutil.move(currPath, os.path.join(finalDir, file))
			
	for folder in folders:
		Loop(os.path.join(foldersDir, folder))

