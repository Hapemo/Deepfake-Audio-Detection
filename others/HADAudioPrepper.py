import os

# Define the directory containing the files
directory = 'C:/Users/jazzt/Documents/GitHub/Deepfake-Audio-Detection/data/HAD/HAD_dev/conbine'  # Change this to your directory path

# Loop through each file in the directory
for filename in os.listdir(directory):
    # Check if the file is a .wav file
    if filename.endswith('.wav'):
        # # Extract the ID and type (fake or real) from the filename
        # parts = filename.split('_')
        # file_type = parts[2]
        # file_id = parts[-1].split('.')[0]  # Get the ID part (without .wav)

        # Create a new name based on the type
        # if 'fake' in file_type:
            # new_name = f'0000000000002{file_id}.wav'
        # elif 'real' in file_type:
            # new_name = f'0000{file_id}.wav'
        # else:
            # Skip files that do not match the expected pattern
            # continue

        if len(filename) > 17:
            new_name = filename[:13] + filename[17:]
        else:
            new_name = filename[4:]

        # Construct full file paths
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_name)

        # Rename the file
        os.rename(old_file, new_file)
        print(f'Renamed: {filename} to {new_name}')