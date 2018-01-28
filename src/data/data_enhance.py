# coding: utf-8
import random
DATA_DIR = "/home/wmq/Desktop/DeepText/SemEval2018Task2/data/emoji"
lowercase = set("abcdefghijklmnopqrstuvwxyz")
stopwords = set("@user i'm it's i a about an are as at be by com for in is it of on or that the this to was with".split())


def strategy_drop(sentence):
    """
    Args:
        sentence: string
    Returns: None or new string by drop out a word
    """
    words = sentence.split()
    if len(words) < 3:
        return None
    drop_id = random.randint(0, len(words)-1)
    sent = []
    for i, word in enumerate(words):
        if i != drop_id:
            sent.append(word)
    return " ".join(sent)


def strategy_remove_punctuation(sentence):
    """
    Args:
        sentence: string
    Returns: None or new string with punctuation removed.
    """
    words = sentence.split()
    sent = []
    for word in words:
        for char in word:
            if char in lowercase:
                sent.append(word)
                break
    if len(sent) != 0:
        return " ".join(sent)
    return None


def strategy_remove_stopwords(sentence):
    """
    Args:
        sentence: string
    Returns: None or new string with stopwords removed.
    """
    words = sentence.split()
    sent = []
    for word in words:
        if word not in stopwords:
            sent.append(word)
    if len(sent) != 0:
        return " ".join(sent)
    return None


def test():
    s = "my grandma is much cooler than yours . 88 years old and getting lit -"
    print(strategy_drop(s))
    print(strategy_remove_punctuation(s))
    print(strategy_remove_stopwords(s))


def main():
    fw = open(DATA_DIR+"/us_train_enhanced.txt", "w")
    raw_n = 0
    new_n = 0
    with open(DATA_DIR+"/us_train_process.txt", "r") as fr:
        for line in fr:
            raw_n += 1
            y, sent = line.strip().split(None, 1)
            s1 = strategy_drop(sent)
            fw.write(line)
            if s1 is not None:
                new_n += 1
                fw.write("%s %s\n" % (y, s1))
            # s2 = strategy_remove_punctuation(sent)
            # if s2 is not None:
            #     new_n += 1
            #     fw.write("%s %s\n" % (y, s2))
            # s3 = strategy_remove_stopwords(sent)
            # if s3 is not None:
            #     new_n += 1
            #     fw.write("%s %s\n" % (y, s3))
    fw.close()
    print("raw_n: %d, new_n: %d" % (raw_n, new_n))

if __name__ == '__main__':
    # test()
    main()
