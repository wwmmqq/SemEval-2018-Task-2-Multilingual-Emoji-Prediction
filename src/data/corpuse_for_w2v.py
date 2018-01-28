# coding: utf-8

""" Sentiment140 eg:

"0","1467810369","Mon Apr 06 22:19:45 PDT 2009","NO_QUERY","_TheSpecialOne_","@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D"
"0","1467810672","Mon Apr 06 22:19:49 PDT 2009","NO_QUERY","scotthamilton","is upset that he can't update his Facebook by texting it... and might cry as a result  School today also. Blah!"
"""
import csv
import sys
sys.path.append('/home/wmq/Desktop/DeepText/SemEval2018Task2/src/data')
from tweet_preprocess import text_process

fw = open('/home/wmq/Desktop/DeepText/SemEval2018Task2/data/Sentiment140/corpus_word.txt', 'w', encoding='utf-8')
in_file = '/home/wmq/Desktop/DeepText/SemEval2018Task2/data/Sentiment140/training.1600000.processed.noemoticon.csv'

with open(in_file, encoding="ISO-8859-1") as csv_file:
    spamreader = csv.reader(csv_file)
    for row in spamreader:
        words = text_process(row[-1])
        line = ' '.join(words)
        # line = ' '.join(line)
        fw.write(line+'\n')

train_file_dir = '/home/wmq/Desktop/DeepText/SemEval2018Task2/data/emoji/data'
train_file = train_file_dir + '/us_train_process.txt'
dev_file = train_file_dir + '/us_test_process.txt'
test_file = train_file_dir + '/us_trial_process.txt'
for in_f in [train_file, dev_file, test_file]:
    with open(in_f, 'r') as fr:
        for line in fr:
            fw.write(' '.join(line.split()[1:]))
fw.close()
