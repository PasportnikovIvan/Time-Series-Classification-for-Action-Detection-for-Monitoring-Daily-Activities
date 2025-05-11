import os
from sklearn.model_selection import train_test_split

def get_file_list(directory, action):
    action_dir = os.path.join(directory, action)
    return [os.path.join(action_dir, file) for file in os.listdir(action_dir) if file.endswith('.json')]

def split_dataset_binary(directory, actions, train_ratio=0.8):
    train_files = []
    test_files = []
    
    for action in actions:
        files = get_file_list(directory, action)
        if not files:
            print(f"No files for {action} in {directory}")
            continue
        train, test = train_test_split(files, train_size=train_ratio, random_state=42)
        train_files.extend(train)
        test_files.extend(test)
    
    return train_files, test_files

def split_dataset(directory, actions, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    train_files = []
    val_files = []
    test_files = []
    
    for action in actions:
        files = get_file_list(directory, action)
        if not files:
            print(f"No files for {action} in {directory}")
            continue
        train, temp = train_test_split(files, train_size=train_ratio, random_state=42)
        val, test = train_test_split(temp, train_size=val_ratio/(val_ratio + test_ratio), random_state=42)
        train_files.extend(train)
        val_files.extend(val)
        test_files.extend(test)
    
    return train_files, val_files, test_files

def save_split_files(train_files, test_files, val_files=None, output_dir="splits"):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train_files.txt"), "w") as f:
        f.write("\n".join(train_files))
    with open(os.path.join(output_dir, "test_files.txt"), "w") as f:
        f.write("\n".join(test_files))
    if val_files:
        with open(os.path.join(output_dir, "val_files.txt"), "w") as f:
            f.write("\n".join(val_files))

actions = ['timed-up-and-go', 'falling', 'sitting', 'standing']
camera_path = "dataset/cameraLandmarks"
global_path = "dataset/globalLandmarks"
dir = "splits/"

# Binary split for global landmarks (train/test only)
train_1, test_1 = split_dataset_binary(global_path, actions)
print("Train files:", len(train_1))
print("Test files:", len(test_1))

# Three-way split for global landmarks (train/val/test)
train_2, val_2, test_2 = split_dataset(global_path, actions)
print("Global Train files:", len(train_2))
print("Global Validation files:", len(val_2))
print("Global Test files:", len(test_2))

save_split_files(train_1, test_1, output_dir=f"{dir}global_tt")
save_split_files(train_2, val_2, test_2, output_dir=f"{dir}global_tvt")