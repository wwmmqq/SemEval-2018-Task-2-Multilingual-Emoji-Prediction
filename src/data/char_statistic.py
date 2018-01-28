# coding: utf-8
import string
in_file = '/home/wmq/Desktop/DeepText/SemEval2018Task2/data/emoji/us_train_process.txt'

chars = set()

with open(in_file, 'r') as fr:
    max_word_len = 0
    cnt = 0
    sum_char_len = 0.0
    sum_sent_len = 0.0
    cnt_sent = 0
    for line in fr:
        sent = line.split('\t\t')[1]
        chars.update(sent)
        ws = sent.split()
        cnt_sent += 1
        sum_sent_len += len(ws)
        for w in ws:
            t = len(w)
            cnt += 1
            sum_char_len += t
            if t > max_word_len:
                max_word_len = t
                print (w)
    print ('max_len_word : %d' % max_word_len)
    print ('avg word len: %0.2f' % (sum_char_len / cnt))
    print ('avg sent len: %0.2f' % (sum_sent_len / cnt_sent))
print ('chars num: %d' % len(chars))

chars_std = string.punctuation + string.ascii_lowercase + string.digits
