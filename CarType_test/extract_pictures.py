import os
import shutil

target_file = './images'
wrong_picture = './wrong_picture'
if not os.path.exists(wrong_picture):
    os.mkdir(wrong_picture)

with open('pre_result.txt', 'r', encoding= 'utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split()[0]
        file_path = os.path.join(target_file, line)
        #shutil.move(file_path, wrong_picture)
        print(file_path)