import gradio as gr
from pyfiles.sg_dataloader import predict
from pyfiles.main import get_model
import torch
import json

CONFIG_PATH = "config/Apple/Apple1.1_eval_sgtts.conf"
model = None
device = None

def LoadModel():
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # # load experiment configurations
    with open(CONFIG_PATH, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    # optim_config = config["optim_config"]
    # optim_config["epochs"] = config["num_epochs"]
    # track = config["track"]
    # assert track in ["SG", "LA", "PA", "DF"], "Invalid track given"
    # if "eval_all_best" not in config:
    #     config["eval_all_best"] = "True"
    # if "freq_aug" not in config:
    #     config["freq_aug"] = "False"

    # make experiment reproducible
    # set_seed(args.seed, config)

    # define database related paths
    # output_dir = Path(args.output_dir)
    # prefix_2019 = "ASVspoof2019.{}".format(track)
    # database_path = Path(config["database_path"])
    # dev_trial_path = None
    # if sgData: # customize here
    #     print("dev_trial_path") 
    # else:
    # dev_trial_path = (database_path /
    #                 "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
    #                     track, prefix_2019))
    # eval_trial_path = None
    # if sgData: # customize here
    #     print("eval_trial_path")
    # else:
    # eval_trial_path = (database_path /
    #                 "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
    #                 track, prefix_2019))

    # if debugPrint:
    #     print("output_dir")
    #     print(output_dir)
    #     print("prefix_2019")
    #     print(prefix_2019)
    #     print("database_path")
    #     print(database_path)
    #     print("dev_trial_path")
    #     print(dev_trial_path)
    #     print("eval_trial_path")
    #     print(eval_trial_path)

    # define model related paths
    # model_tag = "{}_{}_ep{}_bs{}".format(
    #     track,
    #     os.path.splitext(os.path.basename(args.config))[0],
    #     config["num_epochs"], config["batch_size"])
    # if args.comment:
    #     model_tag = model_tag + "_{}".format(args.comment)
    # model_tag = output_dir / model_tag
    # model_save_path = model_tag / "weights"
    # eval_score_path = model_tag / config["eval_output"]
    # traditional_eval_score_path = model_tag / config["traditional_eval_output"]
    # writer = SummaryWriter(model_tag)
    # os.makedirs(model_save_path, exist_ok=True)
    # copy(args.config, model_tag / "config.conf")

    # if debugPrint:
    #     print("model_tag")
    #     print(model_tag)
    #     print("model_save_path")
    #     print(model_save_path)
    #     print("eval_score_path")
    #     print(eval_score_path)
    #     print("writer")
    #     print(writer)

    # set device
    global device
    global model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)

    # if debugPrint:
    #     print("model")
    #     print(model)

    # define dataloaders
    # trn_loader, dev_loader, eval_loader = sg_get_loader(
        # database_path, args.seed, config)
    
    # if debugPrint:
    #     print("trn_loader")
    #     print(trn_loader)
    #     print("dev_loader")
    #     print(dev_loader)
    #     print("eval_loader")
    #     print(eval_loader)

    # evaluates pretrained model and exit script
    model.load_state_dict(
        torch.load(config["model_path"], map_location=device))
    print("Model loaded : {}".format(config["model_path"]))
    # sg_produce_evaluation_file(eval_loader, model, device, eval_score_path)
    # #traditional_evaluation(eval_loader, model, device, traditional_eval_score_path)
    # sg_calculate_EER(cm_scores_file=eval_score_path,
    #                 #    asv_score_file=database_path /
    #                 #    config["asv_score_path"],
    #                     output_file=model_tag / "t-DCF_EER.txt")


def process_audio(audio_file):
    print("audio_file: ", audio_file)
    result = predict("../data/gradioTest/spoof.wav", model, device)
    return result


def main():
    interface = gr.Interface(
        fn=process_audio,                   # Function that processes the audio
        inputs=gr.Audio(type="filepath"),  # Audio input from microphone
        outputs=gr.Textbox(),               # Audio output
        live=True                           # Enable live mode
    )
    
    interface.launch(share = True)


if __name__ == '__main__':
    LoadModel()
    main()