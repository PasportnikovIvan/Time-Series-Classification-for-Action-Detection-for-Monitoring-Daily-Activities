import os
from sklearn.model_selection import train_test_split

def get_file_list(directory, action):
    action_dir = os.path.join(directory, action)
    return [os.path.join(action_dir, file) for file in os.listdir(action_dir) if file.endswith("_repaired.json")]

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

def save_split_files(train_files, val_files, test_files, output_dir="splits"):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train_files.txt"), "w") as f:
        f.write("\n".join(train_files))
    with open(os.path.join(output_dir, "val_files.txt"), "w") as f:
        f.write("\n".join(val_files))
    with open(os.path.join(output_dir, "test_files.txt"), "w") as f:
        f.write("\n".join(test_files))

actions = ['sppb', 'timed-up-and-go', 'falling', 'sitting', 'standing']
camera_train, camera_val, camera_test = split_dataset("../dataset/cameraLandmarks", actions)
global_train, global_val, global_test = split_dataset("../dataset/globalLandmarks", actions)

print("Camera Train files:", len(camera_train))
print("Camera Validation files:", len(camera_val))
print("Camera Test files:", len(camera_test))

save_split_files(camera_train, camera_val, camera_test, "splits/camera")
save_split_files(global_train, global_val, global_test, "splits/global")