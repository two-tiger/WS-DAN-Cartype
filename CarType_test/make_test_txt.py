"""
make txt file for cartype dataset
the txt files include:
1. images.txt
2. image_class_labels.txt
3. train_test_split.txt
"""

from codecs import unicode_escape_decode
import os

images_path = './images/'
class_names = ['car', 'lbus', 'ltruck', 'mbus', 'mpv', 'mtruck', 'nonmotor', 'suv', 'van']

# rename the picture
class_files = os.listdir(images_path)
for class_file in class_files:
    class_path = os.path.join(images_path, class_file)
    fileList = os.listdir(class_path)
    new_name = [class_file + str(i) for i in range(len(fileList))]
    for j in range(len(new_name)):
        os.rename(class_path + '/'+ fileList[j], class_path + '/'+ new_name[j] + '.jpg')

# image.txt
picture_path_list = []
class_files = os.listdir(images_path)
for class_file in class_files:
    file_list = os.listdir(os.path.join(images_path, class_file))
    for file in file_list:
        picture_path_list.append(os.path.join(class_file, file))
with open('images.txt','w',encoding='utf-8') as f:
    for i, picture in enumerate(picture_path_list):
        f.write(str(i+1) + ' ' + picture + '\n')

# class.txt
class_dic = {}
with open('class.txt', 'w') as f:
    for i, class_name in enumerate(class_names):
        f.write(str(i) + ' ' + class_name + '\n')
        class_dic[class_name] = str(i)

# image_class_labels.txt
with open('image_class_labels.txt', 'w') as f:
    for i, picture in enumerate(picture_path_list):
        className, picturePath = picture.split('/')
        f.write(str(i+1) + ' ' + class_dic[className] + '\n')

# train_test_split.txt
seq = [0] * len(picture_path_list)
with open('train_test_split.txt', 'w') as f:
    for i, train_test_index in enumerate(seq):
        f.write(str(i+1)+ ' ' + str(train_test_index) + '\n')
        