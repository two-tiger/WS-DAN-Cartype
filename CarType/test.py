import os

file_path = './CarType/images/mtruck/'
file_list = os.listdir(file_path)
new_name = ['mtruck' + str(i) for i in range(len(file_list))]
for i in range(len(file_list)):
    os.rename(file_path + file_list[i], file_path + new_name[i] + '.jpg')
