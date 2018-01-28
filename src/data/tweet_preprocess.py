# coding: utf-8

# from nltk.tag.stanford import StanfordNERTagger, StanfordPOSTagger
# from nltk.tokenize import StanfordTokenizer
from wordsegment import load, segment
import re
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()

CUR_DIRECTORY = '/home/wmq/Desktop/DeepText/StanfordNLP'
SEGMENT_PATH = CUR_DIRECTORY + '/stanford-segmenter-3.8.0.jar'
NER_MODEL_PATH = CUR_DIRECTORY + '/english.all.3class.distsim.crf.ser.gz'
NER_JAR_PATH = CUR_DIRECTORY + '/stanford-ner.jar'
POS_MODEL_PATH = CUR_DIRECTORY + '/english-left3words-distsim.tagger'
POS_JAR_PATH = CUR_DIRECTORY + '/stanford-postagger.jar'

# ner_tagger = StanfordNERTagger(NER_MODEL_PATH, NER_JAR_PATH, java_options='')
# pos_tagger = StanfordPOSTagger(POS_MODEL_PATH, POS_JAR_PATH, java_options='')
# tokenizer = StanfordTokenizer(SEGMENT_PATH, options={'normalizeOtherBrackets': False,
#                                                      'normalizeParentheses': False})
load()

p_chars = re.compile(r'[^\x20-\x7e]')  # sub of non-ASCII characters
p_keep_chars = re.compile(r"[^a-z!():]")
p_http = re.compile(r'https?://\S+')
p_reduce = re.compile(r"(.)\1{2,}")
p_username = re.compile(r"(^|(?<=[^\w.-]))@[A-Za-z_]+\w+")
p_space = re.compile(r'\s+')  # s -> blank space

g_map = {':)': 'smile', ':-)': 'smile', ':-(': 'bad', '(-:': ''}


def text_process(s):
    s = p_chars.sub('', s)
    s = str.lower(s)
    s = p_http.sub('<url>', s)
    s = p_reduce.sub(r"\1\1", s)
    s = p_username.sub('@user', s)
    s = re.sub(r"&amp(;)?", "&", s)
    s = re.sub(r"/", " ", s)
    words = tknzr.tokenize(s)
    rst = ''
    for w in words:
        if w[0] == '#':
            # rst += ' %s' % w[0]
            rst += ' '
            rst += ' '.join(segment(w))
        else:
            rst += ' %s' % w
    return rst.split()


def remove_puc(s):
    s = p_keep_chars.sub(' ', s)
    return s.split()


def main(suffix='_process'):
    files = ['/home/wmq/Desktop/DeepText/SemEval2018Task2/data/emoji/data/us_trial.txt',
             '/home/wmq/Desktop/DeepText/SemEval2018Task2/data/emoji/data/us_test.txt',
             '/home/wmq/Desktop/DeepText/SemEval2018Task2/data/emoji/data/us_train.txt']
    max_len = 0
    labels = set()
    for f in files:
        f_out = f.split('.')[0] + '%s.txt' % suffix
        fr = open(f, 'r')
        fw = open(f_out, 'w')
        sample_cnt = 0
        for line in fr:
            if sample_cnt % 10000 == 0:
                print ('sample_cnt: %d' % sample_cnt)
            sample_cnt += 1
            t = line.strip().split()
            x = ' '.join(t[1:])
            y = t[0]
            ws = text_process(x)
            # ws = remove_puc(x)
            if len(ws) > max_len:
                max_len = len(ws)
            labels.add(y)
            fw.write('%s ' % y + ' '.join(ws) + '\n')
        print ('finished: %s' % f)
        fr.close()
        fw.close()
        print ('finished sample_cnt: %d' % sample_cnt)
    print ('max sent len : %s' % max_len)
    print ('labels: ' + ' '.join(labels))


if __name__ == '__main__':
    main()
