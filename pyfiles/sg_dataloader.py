# This script will include functions that splits a whole chunk of dataset into train, eval and dev
# Then load them into the system randomized. 
import os
import random
import math
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
    def __init__(self, list_IDs, labels, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list(list_IDs)
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(os.path.join(self.base_dir, key))
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, y

class Dataset_SG_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list(list_IDs)
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(os.path.join(self.base_dir, key))
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key
    

# spoofed data would look something like this 001701010010115.wav, bonafide will look something like this, 01010115.wav
def data_segmenter(dataPath, evalRatio, devRatio):
    print("--- segmenting data ---")
    # eval Ratio is the ratio of the full data used for evaluation
    # dev Ratio is the ratio of the remaining data used for cross validation
    # The segmented data information will be stored in text file in the same directory as dataPath

    # Loop through all the data, keep track of the number of spoofed and bonafide data, add them to a container accordingly.
    bonafide = []
    spoofed = []
    for item in os.listdir(dataPath):
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
    eval = [line + '\n' for line in eval]
    dev = [line + '\n' for line in dev]
    train = [line + '\n' for line in train]
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
                evalData[line] = int(not check_spoof(line))
            elif state == STATES[1]: #dev
                devData[line] = int(not check_spoof(line))
            elif state == STATES[2]: #train
                trainData[line] = int(not check_spoof(line))
    return evalData, devData, trainData

def sg_get_loader(
        database_path: str,
        seed: int,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""
    # track = config["track"]
    # prefix_2019 = "ASVspoof2019.{}".format(track)

    # trn_database_path = database_path / "ASVspoof2019_{}_train/".format(track)
    # dev_database_path = database_path / "ASVspoof2019_{}_dev/".format(track)
    # eval_database_path = database_path / "ASVspoof2019_{}_eval/".format(track)

    # trn_list_path = (database_path /
    #                  "ASVspoof2019_{}_cm_protocols/{}.cm.train.trn.txt".format(
    #                      track, prefix_2019))
    # dev_trial_path = (database_path /
    #                   "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
    #                       track, prefix_2019))
    # eval_trial_path = (
    #     database_path /
    #     "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
    #         track, prefix_2019))

    # d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path,
    #                                         is_train=True,
    #                                         is_eval=False)
    # print("no. training files:", len(file_train))

    segmentInfoPath = os.path.join(database_path, "segment_info.txt")
    # Generate a segment info file if it does not exist
    if not os.path.exists(segmentInfoPath):
        data_segmenter(database_path, 0.2, 0.2) # Should change the ratio according to config, TODO
    
    evalData, devData, trainData = genSpoof_list_sg(segmentInfoPath)

    train_set = Dataset_SG_train(list_IDs=trainData.keys(),
                                 labels=trainData,
                                 base_dir=database_path)
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=gen)

    # _, file_dev = genSpoof_list(dir_meta=dev_trial_path,
    #                             is_train=False,
    #                             is_eval=False)
    # print("no. validation files:", len(file_dev))

    dev_set = Dataset_SG_devNeval(list_IDs=devData.keys(),
                                  base_dir=database_path)
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    # file_eval = genSpoof_list(dir_meta=eval_trial_path,
    #                           is_train=False,
    #                           is_eval=True)
    eval_set = Dataset_SG_devNeval(list_IDs=evalData.keys(),
                                   base_dir=database_path)
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    return trn_loader, dev_loader, eval_loader

def sg_calculate_EER(cm_scores_file,
                    output_file,
                    printout=True):
    # Replace CM scores with your own scores or provide score file as the
    # first argument.
    # cm_scores_file =  'score_cm.txt'

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
    for batch_x, utt_id in data_loader: # running through a loop with a new batch everytime
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x) # Inference with new batch every time
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel() # Calculate the score of a batch 
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist()) # Append the scores of a batch to the master score list

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
    for batch_x, utt_id in data_loader: # running through a loop with a new batch everytime
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x) # Inference with new batch every time 
            # torch.argmax(torch.nn.Softmax(dim=1)(batch_out)).item()
            predicted = [torch.argmax(softmaxed).item() for softmaxed in torch.nn.Softmax(dim=1)(batch_out)]
        # add outputs
        actual_list.extend([0 if check_spoof(id) else 1 for id in utt_id])
        predicted_list.extend(predicted)
    
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

# data_segmenter("data/SG/test", 0.2, 0.2)
# a, b, c = genSpoof_list_sg("data/SG/test/segment_info.txt")

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

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_predict("./data/SG/predictTest", "exp_result/SG_AASIST-sg_ep10_bs4/weights/best.pth", "exp_result/SG_AASIST-sg_ep10_bs4/config.conf", device=device)