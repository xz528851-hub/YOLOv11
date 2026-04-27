import os

folder_path = r"/root/ultralytics-main/roi_line_yolo_dataset_mix/images/val"  # 改成你的文件夹路径

file_count = 0
for item in os.listdir(folder_path):
    full_path = os.path.join(folder_path, item)
    if os.path.isfile(full_path):
        file_count += 1

print(f"文件夹中的文件数目为: {file_count}")