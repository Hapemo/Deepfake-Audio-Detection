''' This script will run through specified folders and perform Silero VAD on all the audio it can find and save it '''
from sg_dataloader import new_data_segmenter, SileroVAD
import argparse
import json
import os
from pathlib import Path
import soundfile as sf
import numpy as np

SAMPLERATE = 16000

def main():
  parser = argparse.ArgumentParser(description="ASVspoof detection system")
  parser.add_argument("--config",
                      dest="config",
                      type=str,
                      help="configuration file",
                      required=True)
  args = parser.parse_args()
  
  # prepare config
  config = None
  with open(args.config, "r") as f_json:
    config = json.loads(f_json.read())

  # Get file names
  segmentInfoPath = os.path.join(Path(config["database_path"]), "segment_info.txt")
  new_data_segmenter(config)

  # run Silero vad on all the files extracted
  counter = 0
  with open(segmentInfoPath, "r") as file:
    for line in file:
      if ".wav" not in line:
        continue

      counter += 1
      if counter % 100 == 0:
        print("Count: ", counter)
      try:
        line = line.rstrip('\n')
        wav = SileroVAD.VAD(line, SAMPLERATE)

        # If you want to change the file path name before saving then do it here
        line = line.replace("/data", "/data/SileroVAD")

        dirPath = os.path.dirname(line)
        if not os.path.exists(dirPath):
          os.makedirs(dirPath)
          print("dirPath created: ", dirPath)
        #---------------------------------------------------------

        sf.write(line, np.ravel(wav), SAMPLERATE)
      except Exception as e:
        print(e)

  






















main()























