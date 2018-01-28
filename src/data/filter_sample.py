# coding: utf-8

train_file = '/home/wmq/Desktop/DeepText/SemEval2018Task2/data/emoji/us_train_process.txt'
dev_file = '/home/wmq/Desktop/DeepText/SemEval2018Task2/data/emoji/us_trial_process.txt'
train_file_new = '/home/wmq/Desktop/DeepText/SemEval2018Task2/data/emoji/us_train_process_new.txt'
dev = set()
with open(dev_file, 'r') as fr:
    for line in fr:
        dev.add(line)

cnt = 0
fw = open(train_file_new, 'w')
with open(train_file, 'r') as fr:
    for line in fr:
        if line in dev:
            cnt += 1
        else:
            fw.write(line)
fw.close()
print ('same sample cnt: %d' % cnt)

#
# src python3 data/filter_sample.py
# same sample cnt: 19901
