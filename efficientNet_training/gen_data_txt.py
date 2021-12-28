# execute command: python gen_data_txt.py -mode train/test/crop
import glob
import argparse
import os


parser = argparse.ArgumentParser(description='generate txt file for YOLOV5 file loading')
arg = parser.parse_args()

if __name__ == '__main__':
    train_ratio = 0.8  # ratio of training data
    content_train = ''
    content_val = ''
    content_total = ''
    path = 'eff_img/*'
    file_name = glob.glob(path)
    train_data_num = int(train_ratio * len(file_name))
    count = 0
    for f in file_name:
        f = os.path.normcase(f)
        print(f)
        if count < train_data_num:
            content_train += (f + '\n')
        else:
            content_val += (f + '\n')
        count += 1
    content_total += (content_train + content_val)
    with open('new_eff_train.txt', 'w') as file:
        file.write(content_train)
        file.close()
    with open('new_eff_val.txt', 'w') as file:
        file.write(content_val)
        file.close()
    with open('new_eff_total.txt', 'w') as file:
        file.write(content_total)
        file.close()