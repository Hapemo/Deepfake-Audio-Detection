# import os

# Hable
# Reads in the protocol list

# protocol_list = None
# def read_protocol(file_path):
#     result = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             columns = line.strip().split()
#             if len(columns) >= 2:
#                 result.append([columns[1], columns[-1]])
#     return result

# # Example usage
# protocol_file_path = 'protocol.txt'
# protocol_list = read_protocol(protocol_file_path)

# counter = 0
# for file in os.listdir("eval"):
#     path = f"eval/{file}"
#     name = path.split(".")[0]
#     nameChanged = False
#     for info in protocol_list:
#         if name in info[0]:
#             spoof = "spoof" in info[1].lower()
#             if spoof:
#                 newname = "1" + f"{counter}".zfill(11) + ".wav"
#             else:
#                 newname = "1" + f"{counter}".zfill(7) + ".wav"
#             os.rename(path, f"eval/{newname}")
#             nameChanged = True
#             counter += 1
#             break
#     if not nameChanged:
#         print(f"Error changing name for {name} in hable")





import os

# Hable
# Reads in the protocol list

protocol_list = None
def read_protocol(file_path):
    result = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            columns = line.strip().split()
            if len(columns) >= 2:
                result.append([columns[1], columns[-1]])
            else:
                print(f"columns failed: {columns}")
    return result

# Example usage
protocol_file_path = 'protocol.txt'
protocol_list = read_protocol(protocol_file_path)
print(len(protocol_list))

counter = 0
for root, dirs, files in os.walk("eval"):
    for file in files:
        if file.endswith(".wav"):
            path = os.path.join(root, file)
            name = os.path.splitext(file)[0]
            nameChanged = False
            for info in protocol_list:
                if name in info[0]:
                    spoof = "spoof" in info[1].lower()
                    if spoof:
                        newname = "1" + f"{counter}".zfill(11) + ".wav"
                    else:
                        newname = "1" + f"{counter}".zfill(7) + ".wav"
                    new_path = os.path.join(root, newname)
                    os.rename(path, new_path)
                    nameChanged = True
                    counter += 1
                    break
            if not nameChanged:
                print(f"Error changing name for {name} in hable")



# Wild
wild = True
import csv

def read_protocol_csv(file_path):
    result = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            if len(row) >= 3:
                result.append([row[0], row[-1]])
    return result

# Example usage
protocol_file_path = 'protocol.csv'
protocol_csv = read_protocol_csv(protocol_file_path)


counter = 0
for root, dirs, files in os.walk("release_in_the_wild"):
    for file in files:
        if file.endswith(".wav"):
            path = os.path.join(root, file)
            name = os.path.splitext(file)[0]
            nameChanged = False
            for info in protocol_list:
                if name in info[0]:
                    spoof = "spoof" in info[1].lower()
                    if spoof:
                        newname = "2" + f"{counter}".zfill(11) + ".wav"
                    else:
                        newname = "2" + f"{counter}".zfill(7) + ".wav"
                    new_path = os.path.join(root, newname)
                    os.rename(path, new_path)
                    nameChanged = True
                    counter += 1
                    break
            if not nameChanged:
                print(f"Error changing name for {name} in hable")


# VCC

# Seq
counter = 0
for root, dirs, files in os.walk("vcc2020-baseline-ref-set"):
    for file in files:
        if file.endswith(".wav"):
            path = os.path.join(root, file)
            newname = "2" + f"{counter}".zfill(11) + ".wav"
            new_path = os.path.join(root, newname)
            os.rename(path, new_path)
            counter += 1

# Cycle VAE
counter = 0
for root, dirs, files in os.walk("reference_v1.0"):
    for file in files:
        if file.endswith(".wav"):
            path = os.path.join(root, file)
            newname = "3" + f"{counter}".zfill(11) + ".wav"
            new_path = os.path.join(root, newname)
            os.rename(path, new_path)
            counter += 1

# 2018
counter = 0
for root, dirs, files in os.walk("VCC20_refs"):
    for file in files:
        if file.endswith(".wav"):
            path = os.path.join(root, file)
            newname = "4" + f"{counter}".zfill(11) + ".wav"
            new_path = os.path.join(root, newname)
            os.rename(path, new_path)
            counter += 1

# bonafide
counter = 0
for root, dirs, files in os.walk("vcc2020_database_evaluation"):
    for file in files:
        if file.endswith(".wav"):
            path = os.path.join(root, file)
            newname = "5" + f"{counter}".zfill(7) + ".wav"
            new_path = os.path.join(root, newname)
            os.rename(path, new_path)
            counter += 1