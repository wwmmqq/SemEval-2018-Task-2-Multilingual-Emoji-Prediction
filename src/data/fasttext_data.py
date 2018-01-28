# coding: utf-8


DATA_DIR = '/home/wmq/Desktop/DeepText/SemEval2018Task2/data/emoji'


def translate(fin, fout):
    with open(fin, 'r') as fr, open(fout, 'w') as fw:
        for line in fr:
            t = line.strip().split('\t\t')
            fw.write('__label__%s %s\n' % (t[0], t[1]))


def translate_y(fin, fout):
    with open(fin, 'r') as fr, open(fout, 'w') as fw:
        for line in fr:
            t = line.strip()
            fw.write('__label__%s\n' % t)

# translate(DATA_DIR+'/us_train_process.txt', DATA_DIR+'/train.txt')
# translate(DATA_DIR+'/us_trial_process.txt', DATA_DIR+'/trial.txt')
translate_y(DATA_DIR+'/us_trial.labels', DATA_DIR+'/trial.labels')

