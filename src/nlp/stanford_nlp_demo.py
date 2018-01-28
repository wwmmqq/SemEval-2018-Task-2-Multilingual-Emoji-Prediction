from nltk.tag.stanford import StanfordNERTagger, StanfordPOSTagger
from nltk.tokenize import StanfordTokenizer
from wordsegment import load, segment

CUR_DIRECTORY = '/home/wmq/Desktop/DeepText/StanfordNLP'
SEGMENT_PATH = CUR_DIRECTORY + '/stanford-segmenter-3.8.0.jar'
NER_MODEL_PATH = CUR_DIRECTORY + '/english.all.3class.distsim.crf.ser.gz'
NER_JAR_PATH = CUR_DIRECTORY + '/stanford-ner.jar'
POS_MODEL_PATH = CUR_DIRECTORY + '/english-left3words-distsim.tagger'
POS_JAR_PATH = CUR_DIRECTORY + '/stanford-postagger.jar'

ner_tagger = StanfordNERTagger(NER_MODEL_PATH, NER_JAR_PATH, java_options='')
pos_tagger = StanfordPOSTagger(POS_MODEL_PATH, POS_JAR_PATH, java_options='')
tokenizer = StanfordTokenizer(SEGMENT_PATH)
load()

s = "@user nah pretty sure it's jackson's great jokes"
ws = tokenizer.tokenize(s)
print(' '.join(ws))
# print (' '.join(segment('#happythankgiving')))
# s = 'i got to to go formal with my best friend @ phi mu at jsu'.split()
# ner_sent = ner_tagger.tag(s)
# pos_sent = pos_tagger.tag(s)
# print (ner_sent)
# print (pos_sent)
