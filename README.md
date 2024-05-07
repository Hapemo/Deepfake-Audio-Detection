
# TensorFlow with GPU acceleration using Windows, Docker and WSL2

This project is built with reference to pendragon AI's example project, modified to use the ASSIST machine learning model. Pendragon is used to get the necessary docker component required for setting up a docker image and container with jupyter notebook. ASSIST is the main machine learning model used for this project. To find out more about pendragon or ASSIST, refer to the other readme on the same directory as this readme.

# Get Started

This project is tested on windows, but it should work on other operating systems. Docker is a prerequisite for this project. To begin, navigate to the directory of this readme file on console, then run this command "docker compose up -d" to build the docker image and compose container.

You might want to consider uploading this directory onto github as a source control, including the data and output folder in gitignore since they would likely take up too much space.

After building, you have to delete the 'sample' from sample.env, so it becomes '.env'. You can always launch the container from docker desktop or using docker commands.
To launch jupyter notebook, on docker desktop, navigate to the container logs to find the jupyter notebook web service url. The host is set to 127.0.0.1 and the port is 8888, but the token is randomized, so the url you need to find would look something like this "http://127.0.0.1:8888/lab?token=df29665c0c8e23d378b1d2f3ad40dea20f0301e79bdbfdee"

The starting directory in the jupyter notebook will be ./notebook/ from here.

# Others

There is a 'others' folder in the current directory. It contains files that does other operations that might be useful.

## .wslconfig

This file is used to limit the amount of memory and cpu WSL2 use on windows. Change the config inside the file to your need, and copy the file into your user directory. To ensure it's the correct directory you pasted in, it might contain folders like '.vscode', 'Desktop', 'Documents', 'Downloads', '.docker'.




## Archive
This command allows you to copy any file or folder in a docker container to your local machine
docker cp <containerId>:/file/path/within/container /host/path/target


# Data Generation
The data consist of 2 main types, bona fit and spoofed data. Bona fide data are actual real speeches recorded from participants, provided by IMDA. Spoofed data are fake speeches generated using Text-To-Speech (TTS) or [Voice Conversion (VC)](Documentations/Voice_Conversion.md). 




