# coding: utf-8

from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer
import re
import string
from wordsegment import load, segment
load()

stemmer = PorterStemmer()
g_divide = '\t\t'
g_chars = ' '+string.punctuation+string.ascii_lowercase+string.digits

tknzr = TweetTokenizer()
p_chars = re.compile(r'[^\x20-\x7e]')  # sub of non-ASCII characters
p_space = re.compile(r'\s+')  # s -> blank space
p_http = re.compile(r'https?://\S+')
p_reduce = re.compile(r"(.)\1{2,}")
p_username = re.compile(r"(^|(?<=[^\w.-]))@[A-Za-z_]+\w+")
p_topic = re.compile(r"(^|(?<=[^\w.-]))#[A-Za-z_]+\w+")
# p_s = re.compile(r"\'s", " \'s")
# p_ve = re.sub(r"\'ve", " \'ve")
# p_not = re.sub(r"n\'t", " n\'t")


def replace_url(text, token='<url>'):    # <LINK>
    """replace http:// or https:// in $text with $token"""
    return p_http.sub(token, text)


def reduce_length(text):
    """
    Replace repeated character sequences of length 3 or greater with sequences
    of length 3.
    """
    return p_reduce.sub(r"\1\1", text)


def remove_username(text):
    """
    Remove Twitter username handles from text.
    """
    return p_username.sub('', text)


def merge_space(text):
    """turn several spaces into one space"""
    return p_space.sub(' ', text)


def text_process(s):
    s = p_chars.sub('', s)
    s = str.lower(s)
    s = p_http.sub('<url>', s)
    s = p_reduce.sub(r"\1\1", s)
    s = p_username.sub('@user', s)
    words = tknzr.tokenize(s)
    rst = ''
    for w in words:
        if w[0] == '#':
            rst += ' '.join(segment(w))
        else:
            rst += ' %s' % w
    return rst.split()


def pre_processing(s):
    s = p_chars.sub('', s)
    s = str.lower(s)
    s = p_http.sub('<url>', s)
    s = p_reduce.sub(r"\1\1", s)
    s = p_username.sub('@user', s)
    return tknzr.tokenize(s)


def pre_processing_2(s):
    s = p_chars.sub('', s)
    s = str.lower(s)
    s = p_http.sub('<url>', s)
    s = p_reduce.sub(r"\1\1", s)
    s = p_username.sub('@user', s)
    words = tknzr.tokenize(s)
    try:
        words = [stemmer.stem(w) for w in words]
    except:
        print (s)

    rst = []
    for w in words:
        if w[0] == '#':
            rst.append(w[1:])
            rst.append(w[1:])
        else:
            rst.append(w)
    return rst


def main1():
    files = [
        ('/home/wmq/Desktop/DeepText/SemEval2018Task2/data/emoji/us_train.text',
         '/home/wmq/Desktop/DeepText/SemEval2018Task2/data/emoji/us_train.labels'),
        ('/home/wmq/Desktop/DeepText/SemEval2018Task2/data/emoji/us_trial.text',
         '/home/wmq/Desktop/DeepText/SemEval2018Task2/data/emoji/us_trial.labels')]
    labels = set()
    max_len = 0
    for fx, fy in files:
        f_out = fx.split('.')[0] + '_process.txt'
        fr_x = open(fx, 'r')
        fr_y = open(fy, 'r')
        fw = open(f_out, 'w')
        sample_cnt = 0
        for x, y in zip(fr_x, fr_y):
            sample_cnt += 1
            ws = pre_processing(x)
            if len(ws) > max_len:
                max_len = len(ws)
            y = y.strip()
            labels.add(y)
            fw.write(y + g_divide + ' '.join(ws) + '\n')
        print ('finished sample_cnt: %d' % sample_cnt)
        fr_x.close()
        fr_y.close()
        fw.close()
    print ('max sent len : %s' % max_len)
    print (labels)


def main(suffix='_process'):
    files = ['/home/wmq/Desktop/DeepText/SemEval2018Task2/data/emoji/us_train.txt',
             '/home/wmq/Desktop/DeepText/SemEval2018Task2/data/emoji/us_trial.txt']
    max_len = 0
    labels = set()
    for f in files:
        f_out = f.split('.')[0] + '%s.txt' % suffix
        fr = open(f, 'r')
        fw = open(f_out, 'w')

        sample_cnt = 0
        for line in fr:
            sample_cnt += 1
            y, x = line.split(g_divide)
            ws = text_process(x)
            if len(ws) > max_len:
                max_len = len(ws)
            labels.add(y)
            # fw.write(y + g_divide + ' '.join(ws) + '\n')
            fw.write('__label__%s ' % y + ' '.join(ws) + '\n')
        fr.close()
        fw.close()
        print ('finished sample_cnt: %d' % sample_cnt)
    print ('max sent len : %s' % max_len)
    print ('labels: ' + ' '.join(labels))


if __name__ == '__main__':

    # s1 = 'www    q!!!!'
    # s2 = '''Cerrando el Ayuntamiento it's a
    # (@ Ayuntamiento de San Sebastián de los Reyes in San Sebastián de los Reyes, Madrid) at https://wwm.jd.com'''
    # s3 = 'Es imposible estudiar cuando alguien te peta mandando 37 audios en serio @user para ya pesa'
    # s4 = "I'm quite happy with the Kindle2."
    # s5 = 'stupid movies ce watched ... mirrors ugggh ... stooopeeed ! ! ! rip off !'
    # print (' '.join(pre_processing(s5)))

    main()
