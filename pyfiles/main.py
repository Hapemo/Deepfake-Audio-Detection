"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA
from sg_dataloader import sg_get_loader, sg_produce_evaluation_file, sg_calculate_EER, traditional_evaluation

from data_utils import (Dataset_ASVspoof2019_train,
                        Dataset_ASVspoof2019_devNeval, genSpoof_list)
# from evaluation import calculate_tDCF_EER
# from evaluation import calculate_EER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)

debugPrint = False
sgData = True

def main(args: argparse.Namespace) -> None:
    print(args)
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]
    assert track in ["SG", "LA", "PA", "DF"], "Invalid track given"
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    prefix_2019 = "ASVspoof2019.{}".format(track)
    database_path = Path(config["database_path"])
    # dev_trial_path = None
    # if sgData: # customize here
    #     print("dev_trial_path") 
    # else:
    dev_trial_path = (database_path /
                    "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                        track, prefix_2019))
    # eval_trial_path = None
    # if sgData: # customize here
    #     print("eval_trial_path")
    # else:
    eval_trial_path = (database_path /
                    "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
                    track, prefix_2019))

    if debugPrint:
        print("output_dir")
        print(output_dir)
        print("prefix_2019")
        print(prefix_2019)
        print("database_path")
        print(database_path)
        print("dev_trial_path")
        print(dev_trial_path)
        print("eval_trial_path")
        print(eval_trial_path)

    # define model related paths
    model_tag = "{}_{}_ep{}_bs{}".format(
        track,
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"], config["batch_size"])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]
    traditional_eval_score_path = model_tag / config["traditional_eval_output"]
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    if debugPrint:
        print("model_tag")
        print(model_tag)
        print("model_save_path")
        print(model_save_path)
        print("eval_score_path")
        print(eval_score_path)
        print("writer")
        print(writer)

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)

    if debugPrint:
        print("model")
        print(model)

    # define dataloaders
    trn_loader, dev_loader, eval_loader = sg_get_loader(
        database_path, args.seed, config)

    if debugPrint:
        print("trn_loader")
        print(trn_loader)
        print("dev_loader")
        print(dev_loader)
        print("eval_loader")
        print(eval_loader)

    # evaluates pretrained model and exit script
    if args.eval:
        model.load_state_dict(
            torch.load(config["model_path"], map_location=device))
        print("Model loaded : {}".format(config["model_path"]))
        print("Start evaluation...")
        sg_produce_evaluation_file(eval_loader, model, device, eval_score_path)
        #traditional_evaluation(eval_loader, model, device, traditional_eval_score_path)
        sg_calculate_EER(cm_scores_file=eval_score_path,
                        #    asv_score_file=database_path /
                        #    config["asv_score_path"],
                           output_file=model_tag / "t-DCF_EER.txt")
        # print("DONE.")
        eval_eer = sg_calculate_EER(
            cm_scores_file=eval_score_path,
            # asv_score_file=database_path / config["asv_score_path"],
            output_file=model_tag/"loaded_model_t-DCF_EER.txt")
        sys.exit(0)

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    if debugPrint:
        print("optim_config")
        print(optim_config)
        print("optimizer")
        print(optimizer)
        print("scheduler")
        print(scheduler)
        print("optimizer_swa")
        print(optimizer_swa)

    best_dev_eer = 1.
    best_eval_eer = 100.
    # best_dev_tdcf = 0.05
    # best_eval_tdcf = 1.
    n_swa_update = 0  # number of snapshots of model to use in SWA
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    if debugPrint:
        print("f_log")
        print(f_log)
        print("metric_path")
        print(metric_path)

    # Training
    for epoch in range(config["num_epochs"]):
        print("Start training epoch{:03d}".format(epoch))
        running_loss = train_epoch(trn_loader, model, optimizer, device,                    # Train the model for one epoch
                                   scheduler, config)
        if debugPrint:
            print("running_loss")
            print(running_loss)
        
        
        # produce_evaluation_file(dev_loader, model, device,                                  # Evaluate the model and save it's evaluation result to dev_score.txt on local device
        #                         metric_path/"dev_score.txt", dev_trial_path)
        sg_produce_evaluation_file(dev_loader, model, device, metric_path/"dev_score.txt")  # Evaluate the model and save it's evaluation result to dev_score.txt on local device

        dev_eer = sg_calculate_EER(                                                            # Calculate the EER based on saved result from dev_score.txt
            cm_scores_file=metric_path/"dev_score.txt",
            # asv_score_file=database_path/config["asv_score_path"],
            output_file=metric_path/"dev_t-DCF_EER_{}epo.txt".format(epoch),
            printout=False)
        print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}".format(                                 # Print current epoch EER and running loss
            running_loss, dev_eer))
        writer.add_scalar("loss", running_loss, epoch)                                      # Saving current epoch EER and running loss to tfevents
        writer.add_scalar("dev_eer", dev_eer, epoch)
        # writer.add_scalar("dev_tdcf", dev_tdcf, epoch)

        # best_dev_tdcf = min(dev_tdcf, best_dev_tdcf)
        if best_dev_eer >= dev_eer:                                                         # Condition for if the best development EER was found
            print("best model find at epoch", epoch)
            best_dev_eer = dev_eer
            torch.save(model.state_dict(),                                                  # Save the model for the best development EER to local device
                       model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))

            # do evaluation whenever best model is renewed
            if str_to_bool(config["eval_all_best"]):
                sg_produce_evaluation_file(eval_loader, model, device, eval_score_path)     # Evaluate the latest best model found based on evaluation dataset
                                        
                eval_eer = sg_calculate_EER(
                    cm_scores_file=eval_score_path,
                    # asv_score_file=database_path / config["asv_score_path"],
                    output_file=metric_path /
                    "t-DCF_EER_{:03d}epo.txt".format(epoch))

                log_text = "epoch{:03d}, ".format(epoch)
                if eval_eer < best_eval_eer:                                                # If the current model achieve best evaluation, update the best model
                    log_text += "best eer, {:.4f}%".format(eval_eer)
                    best_eval_eer = eval_eer
                    torch.save(model.state_dict(),
                               model_save_path / "best.pth")
                # if eval_tdcf < best_eval_tdcf:
                #     log_text += "best tdcf, {:.4f}".format(eval_tdcf)
                #     best_eval_tdcf = eval_tdcf
                if len(log_text) > 0:
                    print(log_text)
                    f_log.write(log_text + "\n")

            print("Saving epoch {} for swa".format(epoch))
            optimizer_swa.update_swa()
            n_swa_update += 1
        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
        # writer.add_scalar("best_dev_tdcf", best_dev_tdcf, epoch)

    print("Start final evaluation")
    epoch += 1
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)
    sg_produce_evaluation_file(eval_loader, model, device, eval_score_path)
    traditional_evaluation(eval_loader, model, device, traditional_eval_score_path)
    eval_eer = sg_calculate_EER(cm_scores_file=eval_score_path,
                                            #  asv_score_file=database_path /
                                            #  config["asv_score_path"],
                                             output_file=model_tag / "t-DCF_EER.txt")
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")
    f_log.write("EER: {:.3f}".format(eval_eer))
    f_log.close()

    torch.save(model.state_dict(),
               model_save_path / "swa.pth")

    if eval_eer <= best_eval_eer:
        best_eval_eer = eval_eer
        torch.save(model.state_dict(),
            model_save_path / "best.pth")
    print("Exp FIN. EER: {:.3f}".format(best_eval_eer))


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"])) # import the module python file from models folder
    _model = getattr(module, "Model") # Get the Model class from the imported module
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model

def get_loader(
        database_path: str,
        seed: int,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""
    track = config["track"]
    prefix_2019 = "ASVspoof2019.{}".format(track)

    trn_database_path = database_path / "ASVspoof2019_{}_train/".format(track)
    dev_database_path = database_path / "ASVspoof2019_{}_dev/".format(track)
    eval_database_path = database_path / "ASVspoof2019_{}_eval/".format(track)

    trn_list_path = (database_path /
                     "ASVspoof2019_{}_cm_protocols/{}.cm.train.trn.txt".format(
                         track, prefix_2019))
    dev_trial_path = (database_path /
                      "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                          track, prefix_2019))
    eval_trial_path = (
        database_path /
        "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
            track, prefix_2019))
    

    d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path,
                                            is_train=True,
                                            is_eval=False)
    if debugPrint:
        print(f"file_train: {file_train}")
        print(f"d_label_trn: {d_label_trn}")
    print("no. training files:", len(file_train))

    train_set = Dataset_ASVspoof2019_train(list_IDs=file_train,
                                           labels=d_label_trn,
                                           base_dir=trn_database_path)
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=gen)

    _, file_dev = genSpoof_list(dir_meta=dev_trial_path,
                                is_train=False,
                                is_eval=False)
    print("no. validation files:", len(file_dev))

    dev_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_dev,
                                            base_dir=dev_database_path)
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    file_eval = genSpoof_list(dir_meta=eval_trial_path,
                              is_train=False,
                              is_eval=True)
    eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval,
                                             base_dir=eval_database_path)
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    return trn_loader, dev_loader, eval_loader


def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str) -> None:
    if debugPrint:
        print("save_path")
        print(save_path)
        print("trial_path")
        print(trial_path)

    """Perform evaluation and save the score to a file"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
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

    if debugPrint:
        print("trial_lines")
        print(trial_lines)
        print("fname_list")
        print(fname_list)
        print("score_list")
        print(score_list)

    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines): # Zipping the name, score and trial line and loop through them
            _, utt_id, _, src, key = trl.strip().split(' ')
            assert fn == utt_id # Ensuring the data info from trial lines matches the fname list
            fh.write("{} {} {} {}\n".format(utt_id, src, key, sco)) # Saving the data to another file, cumulative of the evaluation result in the form of (LA_D_1047731 - bonafide -0.00691329687833786)
    print("Scores saved to {}".format(save_path))


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train() # Setting model to train mode

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_y in trn_loader: # Getting each batch of data for training
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"])) # Doing inference with model
        # print(f"batch_x: {batch_x}, \nbatch_out: {batch_out}, \nsoftmax:{nn.Softmax(dim=1)(batch_out)}, \nbatch_y: {batch_y}")
        batch_loss = criterion(batch_out, batch_y) # Calculating batch loss
        running_loss += batch_loss.item() * batch_size # Accumulating batch losses
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    main(parser.parse_args())
