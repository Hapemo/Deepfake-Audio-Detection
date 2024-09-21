# This script will include functions that splits a whole chunk of dataset into train, eval and dev
# Then load them into the system randomized. 
import os
import random
import math
from re import L
from typing import List
import soundfile as sf
from torch.utils.data import DataLoader, Dataset
from data_utils import pad, pad_random, genSpoof_list
import torch
from torch import Tensor
from utils import seed_worker
import numpy as np
from evaluation import compute_eer
from importlib import import_module
import json

# 12 is the normal number of characters for a bonafide data
# Returns true when it's spoof
def check_spoof(filename):
    return len(filename) > 12 

class Dataset_SG_train(Dataset):
    def __init__(self, list_IDs, labels):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list(list_IDs)
        self.labels = labels
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(key)
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, y

class Dataset_SG_devNeval(Dataset):
    def __init__(self, list_IDs):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list(list_IDs)
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(key)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, os.path.basename(key)
    

# spoofed data would look something like this 001701010010115.wav, bonafide will look something like this, 01010115.wav
def data_segmenter(dataPath, evalRatio, devRatio):
    print("--- segmenting data ---")
    # eval Ratio is the ratio of the full data used for evaluation
    # dev Ratio is the ratio of the remaining data used for cross validation
    # The segmented data information will be stored in text file in the same directory as dataPath

    # Loop through all the data, keep track of the number of spoofed and bonafide data, add them to a container accordingly.
    bonafide = []
    spoofed = []
    # config to stop
    items = random.shuffle(os.listdir(dataPath))
    for item in items:
        if ".wav" not in item.lower():
            continue
        if check_spoof(item): # 12 is the normal number of characters for a bonafide data
            spoofed.append(item)
        else:
            bonafide.append(item)
    print(f"bonafide Count: {len(bonafide)}")
    print(f"spoofed Count: {len(spoofed)}")
    random.shuffle(bonafide)
    random.shuffle(spoofed)
    evalBonafideCount = math.floor(len(bonafide) * evalRatio)
    evalSpoofCount = math.floor(len(spoofed) * evalRatio)
    devBonafideCount = math.floor((len(bonafide)-evalBonafideCount) * devRatio)
    devSpoofCount = math.floor((len(spoofed)-evalSpoofCount) * devRatio)
    
    # Open a output file and start writing the names to it. The format will be eval first, followed by dev then train, each section separated by an additional newline.
    # Extracting the data for eval and dev
    eval = bonafide[:evalBonafideCount]
    eval.extend(spoofed[:evalSpoofCount])
    bonafide = bonafide[evalBonafideCount:]
    spoofed = spoofed[evalSpoofCount:]
    dev = bonafide[:devBonafideCount]
    dev.extend(spoofed[:devSpoofCount])
    bonafide = bonafide[devBonafideCount:]
    spoofed = spoofed[devSpoofCount:]
    train = bonafide + spoofed
    print(f"eval Count: {len(eval)}")
    print(f"dev Count: {len(dev)}")
    print(f"train Count: {len(train)}")
    random.shuffle(eval)
    random.shuffle(dev)
    random.shuffle(train)

    # Writing it to the output file
    eval = [os.path.join(dataPath, line) + '\n' for line in eval]
    dev = [os.path.join(dataPath, line) + '\n' for line in dev]
    train = [os.path.join(dataPath, line) + '\n' for line in train]
    eval.insert(0,"eval\n")
    dev.insert(0,"dev\n")
    train.insert(0,"train\n")
    
    filename = "segment_info.txt"
    with open (os.path.join(dataPath, filename), 'w') as outputFile:
        outputFile.writelines(eval)
        outputFile.writelines(dev)
        outputFile.writelines(train)

# data_segmenter("C:/Users/jazzt/Desktop/CCA/Deepfake_Audio_Detection/Mangio-RVC-v23.7.0_INFER_TRAIN/Mangio-RVC/opt/Mangio_RVC_spoofed_data",
#                0.2, 0.2)

def find_folders(path, collectedList):
    """
    Recursively finds and adds all subfolder names to result_list.
    
    Args:
    target_folder (str): The name of the target folder to find.
    path (str): The current directory path to search in.
    result_list (list): The list to store subfolder names.
    """
    print(f"Adding everything in {path}")
    for root, dirs, files in os.walk(path):
        for file in files:
            collectedList.append(os.path.join(root, file))
        for folder_name in dirs:
            folder_path = os.path.join(root, folder_name)
            # Recursively search in subfolders
            find_folders(folder_path, collectedList)
        # No need to continue once we've traversed the target directory
        break

def AddTargetedFiles(target_folders: list, path: str, collectedList: list, blacklist_folders: list):
    """
    Recursively searches for the target folder and its subfolders.
    
    Args:
    target_folders (list): The name of the target folder to find.
    path (str): The current directory path to search in.
    result_list (list): The list to store subfolder names.
    """
    for item in os.listdir(path):
        if len(target_folders) == 0:
            return
        
        item_path = os.path.join(path, item)
        if item in blacklist_folders:
            continue
        if os.path.isdir(item_path):
            if item in target_folders:
                find_folders(item_path, collectedList)
                target_folders.remove(item)
            else:
                print(f"entering folder: {item}")
                AddTargetedFiles(target_folders, item_path, collectedList, blacklist_folders)

def PrintDataStats(title, data):
    spoofCount = 0
    for name in data:
        name = os.path.basename(name)
        if check_spoof(name):
            spoofCount += 1
    print(f"{title} spoof count: {spoofCount}")
    print(f"{title} bonafide count: {len(data) - spoofCount}")
    print(f"{title} total count: {len(data)}")

def RemoveSameTargetSource(dataset: list):
    temp = dataset.copy()
    for data in dataset:
        name = os.path.basename(data)
        if check_spoof(name) and (name[0:4] == name[4:8]):
            temp.remove(data)
    return temp


# spoofed data would look something like this 001701010010115.wav, bonafide will look something like this, 01010115.wav
def new_data_segmenter(config): # config["eval_folders"] or database_path or dev_folders or train_folders
    print("--- segmenting data ---")
    
    # Initializing the data path and folders
    database_path = config["database_path"]
    eval_folders = config["eval_folders"]
    dev_folders = config["dev_folders"]
    train_folders = config["train_folders"]
    blacklist_folders = config["blacklist_folders"]

    eval_folders = [item.strip() for item in eval_folders.split(',')]
    dev_folders = [item.strip() for item in dev_folders.split(',')]
    train_folders = [item.strip() for item in train_folders.split(',')]
    blacklist_folders = [item.strip() for item in blacklist_folders.split(',')]

    eval_files = []
    dev_files = []
    train_files = []

    # Loop through all the data, add relevants ones to train eval and dev
    AddTargetedFiles(eval_folders, database_path, eval_files, blacklist_folders)
    AddTargetedFiles(dev_folders, database_path, dev_files, blacklist_folders)
    AddTargetedFiles(train_folders, database_path, train_files, blacklist_folders)

    # Remove same target source speaker for VC spoof data
    if config["remove_same_source_target_speaker"]:
        eval_files = RemoveSameTargetSource(eval_files)
        dev_files = RemoveSameTargetSource(dev_files)
        train_files = RemoveSameTargetSource(train_files)

    # Count the stats of the data
    PrintDataStats("Eval", eval_files)
    PrintDataStats("Dev", dev_files)
    PrintDataStats("Train", train_files)

    # Loop through all the data, keep track of the number of spoofed and bonafide data, add them to a container accordingly.
    
    # Open a output file and start writing the names to it. The format will be eval first, followed by dev then train, each section separated by an additional newline.
    # Extracting the data for eval and dev

    eval_files = [line + '\n' for line in eval_files]
    dev_files = [line + '\n' for line in dev_files]
    train_files = [line + '\n' for line in train_files]

    # Writing it to the output file
    eval_files.insert(0,"eval\n")
    dev_files.insert(0,"dev\n")
    train_files.insert(0,"train\n")
    
    filename = "segment_info.txt"
    with open (os.path.join(database_path, filename), 'w') as outputFile:
        outputFile.writelines(eval_files)
        outputFile.writelines(dev_files)
        outputFile.writelines(train_files)


def genSpoof_list_sg(segmentedFile):
    STATES = ["eval","dev","train"]
    evalData = {}
    devData = {}
    trainData = {}
    with open(segmentedFile, "r") as file:
        for line in file:
            line = line.rstrip('\n')
            if line in STATES:
                state = line
                continue
            if state == STATES[0]: #eval
                evalData[line] = int(not check_spoof(os.path.basename(line)))
            elif state == STATES[1]: #dev
                devData[line] = int(not check_spoof(os.path.basename(line)))
            elif state == STATES[2]: #train
                trainData[line] = int(not check_spoof(os.path.basename(line)))
    return evalData, devData, trainData

def sg_get_loader(
        database_path: str,
        seed: int,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""

    segmentInfoPath = os.path.join(database_path, "segment_info.txt")
    # Generate a segment info file if it does not exist
    if not os.path.exists(segmentInfoPath):
        if bool(int(config["use_new_fileloader"])):
            new_data_segmenter(config)
        else:
            data_segmenter(database_path, 0.2, 0.2) # Should change the ratio according to config, TODO
    
    evalData, devData, trainData = genSpoof_list_sg(segmentInfoPath)

    train_set = Dataset_SG_train(list_IDs=trainData.keys(),
                                 labels=trainData)
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=gen)

    dev_set = Dataset_SG_devNeval(list_IDs=devData.keys())
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    eval_set = Dataset_SG_devNeval(list_IDs=evalData.keys())
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    return trn_loader, dev_loader, eval_loader

def sg_calculate_EER(cm_scores_file,
                    output_file,
                    printout=True):
    # Load CM scores
    cm_data = np.genfromtxt(cm_scores_file, dtype=str)
    # cm_utt_id = cm_data[:, 0]
    cm_keys = cm_data[:, 1]
    cm_scores = cm_data[:, 2].astype(np.float64)

    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']

    # if debugPrint:
    #     print("cm_keys")
    #     print(cm_keys)
    #     print("cm_scores")
    #     print(cm_scores)
    #     print("bona_cm")
    #     print(bona_cm)
    #     print("spoof_cm")
    #     print(spoof_cm)

    # EERs of the standalone systems and fix ASV operating point to
    # EER threshold
    # eer_asv, asv_threshold = compute_eer(tar_asv, non_asv)
    eer_cm = compute_eer(bona_cm, spoof_cm)[0]

    if printout:
        with open(output_file, "w") as f_res:
            f_res.write('\nCM SYSTEM\n')
            f_res.write('\tEER\t\t= {:8.9f} % '
                        '(Equal error rate for countermeasure)\n'.format(
                            eer_cm * 100))

        os.system(f"cat {output_file}")
    
    return eer_cm * 100

def sg_produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    fname_list = []
    score_list = []
    dataCount = 0
    for batch_x, utt_id in data_loader: # running through a loop with a new batch everytime
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x) # Inference with new batch every time
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel() # Calculate the score of a batch 
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist()) # Append the scores of a batch to the master score list
        dataCount += len(utt_id)
        print(f"Eval {dataCount} data finished")

    # if debugPrint:
    #     print("fname_list")
    #     print(fname_list)
    #     print("score_list")
    #     print(score_list)

    assert len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco in zip(fname_list, score_list): # Zipping the name, score and trial line and loop through them
            # _, utt_id, _, src, key = trl.strip().split(' ')
            fh.write("{} {} {}\n".format(fn, ("spoof" if check_spoof(fn) else "bonafide"), sco)) # Saving the data to another file, cumulative of the evaluation result in the form of (LA_D_1047731 - bonafide -0.00691329687833786)
    print("Scores saved to {}".format(save_path))

def traditional_evaluation(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str) -> None:
    """Perform a traditional evaluation and save the score to a file"""
    # Evaluation 
    model.eval()
    actual_list = []
    predicted_list = []
    dataCount = 0
    for batch_x, utt_id in data_loader: # running through a loop with a new batch everytime
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x) # Inference with new batch every time 
            # torch.argmax(torch.nn.Softmax(dim=1)(batch_out)).item()
            predicted = [torch.argmax(softmaxed).item() for softmaxed in torch.nn.Softmax(dim=1)(batch_out)]
        # add outputs
        actual_list.extend([0 if check_spoof(id) else 1 for id in utt_id])
        predicted_list.extend(predicted)
        dataCount += len(utt_id)
        print(f"Eval {dataCount} data finished")
    
    # False positive, false negative, true positive, true negative
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for act, pred in zip(actual_list, predicted_list):
        if act:
            if pred:
                TP += 1
            else: 
                FN += 1
        else:
            if pred:
                FP += 1
            else: 
                TN += 1
    
    # Sensitivity, Specificity, Positive Predictive Value, Negative Predictive Value
    sensitivity = float("Nan") if (TP + FN) == 0 else TP/(TP + FN)
    specificity = float("Nan") if (TN + FP) == 0 else TN/(TN + FP)
    PPV = float("Nan") if (TP + FP) == 0 else TP/(TP + FP)
    NPV = float("Nan") if (TN + FN) == 0 else TN/(TN + FN)
    
    # Prediction and actual counting data
    predicted_list.count(0)
    predicted_list.count(1)
    actual_list.count(0)
    actual_list.count(1)
    prediction_score = (TP+TN) / len(actual_list)

    # Writing to external file
    with open(save_path, "w") as fh:
        fh.write(f"--- Evaluation Result ---\n\
Predicted Spoof: {predicted_list.count(0)}\n\
Predicted Bonafide: {predicted_list.count(1)}\n\
Actual Spoof: {actual_list.count(0)}\n\
Actual Bonafide: {actual_list.count(1)}\n\
Prediction Score: {prediction_score}\n\
TP: {TP}\n\
FP: {FP}\n\
FN: {FN}\n\
TN: {TN}\n\
sensitivity: {sensitivity}\n\
specificity: {specificity}\n\
PPV: {PPV}\n\
NPV: {NPV}\n\
")
    print("Scores saved to {}".format(save_path))

def predict(
    dataPath: str,
    model,
    device: torch.device) -> None:
    """Perform a traditional evaluation and save the score to a file"""
    # Evaluation 
    model.eval()

    # Load in data
    X, _ = sf.read(dataPath)
    X_pad = pad(X, 64600) # take ~4 sec audio (64600 samples)
    x_inp = Tensor([X_pad])

    x_inp = x_inp.to(device)
    pred = "Error"
    with torch.no_grad():
        _, batch_out = model(x_inp) # Inference with new batch every time 
        # torch.argmax(torch.nn.Softmax(dim=1)(batch_out)).item()
        val1 = torch.nn.Softmax(dim=1)(batch_out)
        pred = torch.argmax(val1).item()
        print("val1: ", val1)
        print("pred: ", pred)
    return pred

def custom_predict(speech, model, device: torch.device, audioLength = 4, sampleRate = 16000) -> None:
    """Perform a traditional evaluation and save the score to a file"""
    # Evaluation 
    model.eval()

    # Load in data
    if type(speech) == str: X, sampleRate = sf.read(speech)
    else: X = speech
    
    X_pad = pad(X, audioLength * sampleRate) # take ~4 sec audio (64600 samples)
    x_inp = Tensor([X_pad])

    x_inp = x_inp.to(device)
    pred = "Error"
    with torch.no_grad():
        _, batch_out = model(x_inp) # Inference with new batch every time 
        # torch.argmax(torch.nn.Softmax(dim=1)(batch_out)).item()
        val1 = torch.nn.Softmax(dim=1)(batch_out)
        pred = torch.argmax(val1).item()
    return pred


def batch_predict(
    datapath,
    modelpath,
    configpath,
    device: torch.device) -> None:
    """Perform prediction on a dataset and prints out in console"""
    # Load config
    with open(configpath, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]

    # Loop through folder and gather information on the data
    nameList = os.listdir(datapath)

    predict_set = Dataset_SG_devNeval(list_IDs=nameList, base_dir=datapath)
    predict_loader = DataLoader(predict_set,
                                batch_size=config["batch_size"],
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)
    
    # Loading Model
    module = import_module("models.{}".format(model_config["architecture"])) # import the module python file from models folder
    model = module.Model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))
    model.load_state_dict(torch.load(modelpath))

    # Evaluation 
    model.eval()
    id_list = []
    actual_list = []
    predicted_list = []
    for batch_x, utt_id in predict_loader: # running through a loop with a new batch everytime
        print(utt_id)
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x) # Inference with new batch every time 
            # torch.argmax(torch.nn.Softmax(dim=1)(batch_out)).item()
            predicted = [torch.argmax(softmaxed).item() for softmaxed in torch.nn.Softmax(dim=1)(batch_out)]
        # add outputs
        actual_list.extend([0 if check_spoof(id) else 1 for id in utt_id])
        predicted_list.extend(predicted)
        id_list.extend(utt_id)
    
    # Calculating accuracy
    accuracy = 0
    for act, pred in zip(actual_list, predicted_list):
        if act == pred:
            accuracy += 1
    accuracy /= len(actual_list) * 100

    # Printing result
    print(f"accuracy: {accuracy:.2f}%")
    for id, pred in zip(id_list, predicted_list):
        print(f"{id}: {pred}")

# device = "cuda" if torch.cuda.is_available() else "cpu"
# batch_predict("./data/SG/predictTest", "exp_result/SG_AASIST-sg_ep10_bs4/weights/best.pth", "exp_result/SG_AASIST-sg_ep10_bs4/config.conf", device=device)