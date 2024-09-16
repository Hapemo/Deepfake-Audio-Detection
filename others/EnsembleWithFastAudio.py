''' This python script takes in 2 evaluation score text files, one generated from AASIST and the other generated from FastAudio. 
It will normalize the logit scores first before calculating the weighted average, finally apply EER calculation on them. 
NOTE: The method to differentiate between spoof and bonafide is purely based on the naming. More than 8 characters in the name means it's spoof.

'''
import numpy as np

FAScoreFile = ""
AASISTScoreFile = ""
AASISTWEIGHT = 0.2

def EnsembleEER(FAScoreFile:str, AASISTScoreFile:str):
    FAFile = open(FAScoreFile, "r")
    AASISTFile = open(AASISTScoreFile, "r")

    FALogits = ExtractLogit(False, FAFile)
    AASISTLogits = ExtractLogit(True, AASISTFile)

    FALogits = MinMaxNorm(FALogits)
    AASISTLogits = MinMaxNorm(AASISTLogits)

    ensembledLogits = EnsembleLogits(AASISTLogits, FALogits, AASISTWEIGHT)

    targetList, nontargetList = SplitTargetNonTarget(ensembledLogits)

    print(f"targetList: {targetList}, nontargetList: {nontargetList}")

    eer, something = compute_eer(np.array(targetList), np.array(nontargetList))

    print(f"EER: {eer}")
    print(f"something: {something}")


def ExtractLogit(AASIST: bool, file):
    returndict = {}
    for line in file.readlines():
        line.replace('\n','')

        if AASIST:
            infoList = line.split(" ")
            name = infoList[0][:-4]# Assuming this is a wav file
            logit = infoList[2]
            returndict[name] = float(logit)
        else:
            infoList = line.split(" ")
            returndict[infoList[0]] = float(infoList[1])

    return returndict

def MinMaxNorm(logitDict: dict):
    ''' normalize the logits in a dictionary '''
    logits = list(logitDict.values())
    min_logit = min(logits)
    max_logit = max(logits)
    for key in logitDict:
        logitDict[key] = (logitDict[key] - min_logit) / (max_logit - min_logit)
    return logitDict

def EnsembleLogits(l0:dict, l1:dict, l0Weight) -> dict:
    ''' Ensemble the logits together and return a new dictionary '''
    l1Weight = 1 - l0Weight
    if l0Weight > 1 or l0Weight < 0:
        print("l0Weight is not between 0 and 1, it is ", l0Weight)
        return {}

    if len(l0) != len(l1):
        print(f"l0 and l1 has different amount of items! l0: {l0} and l1: {l1}")
        return {}
    
    ensembledLogits = {}
    for key in l0:
        if key not in l1:
            print(f"Error: {key} exists in l0 but not l1")
            return {}

        ensembledLogits[key] = l0[key]*l0Weight + l1[key]*l1Weight
    
    return ensembledLogits

def SplitTargetNonTarget(combined:dict):
    targetList = []
    nontargetList = []
    for key in combined:
        if IsSpoof(key):
            nontargetList.append(combined[key])
        else:
            targetList.append(combined[key])
    return targetList, nontargetList

def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - \
        (np.arange(1, n_scores + 1) - tar_trial_sums)

    # false rejection rates
    frr = np.concatenate(
        (np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums /
                          nontarget_scores.size))  # false acceptance rates
    # Thresholds are the sorted scores
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds

def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]

def IsSpoof(name:str):
    return len(name) > 8


EnsembleEER(FAScoreFile, AASISTScoreFile)





