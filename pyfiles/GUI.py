import gradio as gr
from sg_dataloader import predict
from main import get_model
import torch
import json

CONFIG_PATH = "../config/Apple/GUITest.conf"
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

    # set device
    global device
    global model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)

    # evaluates pretrained model and exit script
    model.load_state_dict(
        torch.load(config["model_path"], map_location=device))
    print("Model loaded : {}".format(config["model_path"]))


def process_audio(audio_file):
    print("audio_file: ", audio_file)
    result = "Bonafide" if predict(audio_file, model, device) else "Spoof"
    return result


def main():
    interface = gr.Interface(
        fn=process_audio,                   # Function that processes the audio
        inputs=gr.Audio(type="filepath"),  # Audio input from microphone
        outputs=gr.Textbox(),               # Audio output
        live=True                           # Enable live mode
    )
    
    interface.launch(share = True, server_port = 7860)


if __name__ == '__main__':
    LoadModel()
    main()