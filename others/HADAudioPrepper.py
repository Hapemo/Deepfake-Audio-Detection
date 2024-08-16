'''
This file reads the metafile of the data and copys the files indicated in it from one directory to another.
Depending on the tag, it filters out specific files. (like pure spoof/bonafide or not)
Change the directory, metafilepath, outputDir and ignore condition accordingly
'''
import os
import shutil

# Define the directory containing the files
directory = 'C:/Users/jazzt/Documents/GitHub/Deepfake-Audio-Detection/data/HAD_dev/OG'  # Change this to your directory path
metafilepath = 'C:/Users/jazzt/Documents/GitHub/Deepfake-Audio-Detection/data/HAD_dev/HAD_dev_label.txt'
outputDir = 'C:/Users/jazzt/Documents/GitHub/Deepfake-Audio-Detection/data/HAD_dev/Pure'

file = open(metafilepath)
for line in file.readlines():
    line = line.strip()
    parts = line.split(" ")

    name = parts[0]
    tag = parts[1]

    # Set your condition to ignore here 
    if ('/' in tag):
        continue

    print(line)
    # ---------------------------------

    source = os.path.join(directory, name) + ".wav"

    # Extract the ID and type (fake or real) from the filename
    file_id = name.split('_')[-1]

    # Create a new name based on the type
    if 'fake' in name:
        new_name = f'000000002{file_id}.wav'
    elif 'real' in name:
        new_name = f'{file_id}.wav'
    else:
        # Skip files that do not match the expected pattern
        continue

    destination = os.path.join(outputDir, new_name)
    
    shutil.copy(source, destination)

# Loop through each file in the directory
# for filename in os.listdir(directory):
#     # Check if the file is a .wav file
#     if filename.endswith('.wav'):
#         # Extract the ID and type (fake or real) from the filename
#         parts = filename.split('_')
#         file_type = parts[2]
#         file_id = parts[-1].split('.')[0]  # Get the ID part (without .wav)

#         # Create a new name based on the type
#         if 'fake' in file_type:
#             new_name = f'0000000000002{file_id}.wav'
#         elif 'real' in file_type:
#             new_name = f'0000{file_id}.wav'
#         else:
#             # Skip files that do not match the expected pattern
#             continue

#         # Construct full file paths
#         old_file = os.path.join(directory, filename)
#         new_file = os.path.join(directory, new_name)

#         # Rename the file
#         os.rename(old_file, new_file)
#         print(f'Renamed: {filename} to {new_name}')