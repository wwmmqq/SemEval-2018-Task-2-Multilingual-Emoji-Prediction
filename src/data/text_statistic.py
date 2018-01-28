from data_utils import load_word_embedding_std

data_dir = '/home/wmq/Desktop/DeepText/SemEval2018Task2/data/embed'


def char_info():
    word2id, word_embed = load_word_embedding_std(
        200,
        data_dir+'/tweet_we.bin')
    print (word_embed.shape)
    words = sorted(word2id.keys(), key=lambda x: len(x))

    cnt = 0
    cnt_5 = 0.0
    cnt_10 = 0.0
    cnt_15 = 0.0
    cnt_20 = 0.0
    lease = 0.0
    with open(data_dir+'/word_list.txt', 'w') as fw:
        for w in words:
            fw.write('%s\n' % w)
            w_l = len(w)
            cnt += w_l
            if w_l < 21:
                cnt_20 += 1

                if w_l < 16:
                    cnt_15 += 1
                    if w_l < 11:
                        cnt_10 += 1
                        if w_l < 6:
                            cnt_5 += 1
            else:
                lease += 1

        print (cnt / len(words))

    print ('5 : %0.3f' % (cnt_5/len(words)))
    print ('10 : %0.3f' % (cnt_10/len(words)))
    print ('15 : %0.3f' % (cnt_15/len(words)))
    print ('20 : %0.3f' % (cnt_20/len(words)))
    # idx: 79559
    # ce
    # shape: (79559, 200)
    # (79559, 200)
    # 7.013813647733129
    # 5: 0.338
    # 10: 0.888
    # 15: 0.985
    # 20: 0.998


def sent_info():
    cnt = 0.0
    len10 = 0.0
    len15 = 0.0
    len20 = 0.0
    len30 = 0.0
    len40 = 0.0
    with open('/home/wmq/Desktop/DeepText/SemEval2018Task2/data/emoji/us_train_process.txt',
              'r') as fr:
        for line in fr:
            ws = line.split('\t\t')[1].split()
            t = len(ws)
            cnt += 1
            if t < 41:
                len40 += 1
                if t < 31:
                    len30 += 1
                    if t < 21:
                        len20 += 1
                        if t < 16:
                            len15 += 1
                            if t < 11:
                                len10 += 1

        print (""" sent infor:
         len < 10: {}
         len < 15: {}
         len < 20: {}
         len < 30: {}
         len < 40: {}
        """.format(len10 / cnt,
                   len15 / cnt,
                   len20 / cnt,
                   len30 / cnt,
                   len40 / cnt,
                   ))


sent_info()

# sent infor:
# len < 10: 0.348135028794
# len < 15: 0.737266547178
# len < 20: 0.952447211293
# len < 30: 0.999391246248
# len < 40: 0.999995757814

