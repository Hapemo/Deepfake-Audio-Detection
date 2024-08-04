import gradio as gr
from sg_dataloader import predict
from main import get_model
import torch
import json
import os

CONFIG_PATH = "GUITest.conf"
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

def process_audio(audio_file, selected_file):
    file = ""
    if audio_file is not None:
        file = audio_file
    elif selected_file is not None:
        file = selected_file
    else:
        print("No audio file selected")
        return "No audio file selected"
    
    print("audio_file: ", file)
    result = "Bonafide" if predict(file, model, device) else "Spoof"
    return file, result


def main():
    # Prepare audio files
    audio_dir = "audio"
    audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if os.path.isfile(os.path.join(audio_dir, f))]

    print("audiofiles: ", audio_files)

    interface = gr.Interface(
        fn=process_audio,                   # Function that processes the audio
        inputs=[
            gr.Audio(type="filepath", label="Upload Audio"),  # Audio input from microphone
            gr.Dropdown(audio_files, label="Select Audio")    # Dropdown list of audio files
        ],
        outputs=[
            gr.Audio(type="filepath"), 
            gr.Textbox()
        ],
        live=True,                           # Enable live mode
        title="Deepfake Audio Detection",  # Add a title
        description="Upload an audio file or select one from the dropdown list to process it. Model is not 100 percent accurate, for example, whisperSpeech_2 in dropdown list should be spoof but model detected bonafide."  # Add a description
    )
    
    interface.launch(share = True, server_name = "0.0.0.0", server_port = 7860)


if __name__ == '__main__':
    LoadModel()
    main()