# coding: utf-8
import numpy as np

g_label2id = {'__label__%d' % i: i for i in range(20)}
out_file = "/home/wmq/Desktop/DeepText/SemEval2018Task2/result/submit/english.output.txt"

result_files = [
    "/home/wmq/Desktop/DeepText/SemEval2018Task2/result/submit/fasttext_enhance.txt",
    "/home/wmq/Desktop/DeepText/SemEval2018Task2/result/submit/nbow_25000.txt",
    "/home/wmq/Desktop/DeepText/SemEval2018Task2/result/submit/char_gate_word_att_enhance_big_122000.txt",
    "/home/wmq/Desktop/DeepText/SemEval2018Task2/result/submit/char_gate_word_att_enhance_small_69000.txt"
]


def get_all_result(files, save=False):
    results = []

    for f in files:
        y = []
        with open(f, 'r') as fr:
            for line in fr:
                y.append(g_label2id[line.strip()])
        results.append(y)

    results = np.array(results, dtype=np.int32)
    print(results.shape)

    if save:
        all_result_file = "y_all.txt"
        with open(all_result_file, "w") as fw:
            for i in range(results.shape[1]):
                fw.write("%d %d %d %d\n" % (
                    results[0][i], results[1][i], results[2][i], results[3][i]))
    return results


def single(in_file):
    result_file = "single_english.output.txt"
    fw = open(result_file, "w")
    with open(in_file, 'r') as fr:
        for line in fr:
            fw.write("%d\n" % g_label2id[line.strip()])


def ensemble():
    results = get_all_result(result_files)
    write_file = "english.output.txt"
    fw = open(write_file, 'w')
    for i in range(results.shape[1]):
        y = results[-1][i]
        a = results[0][i]
        b = results[1][i]
        c = results[2][i]

        if y != c:
            if a == c and b == c:
                y = c
            elif (a == c and a != y) or (b == c and b != y):
                    y = c
            elif (a == b) and (y != a) and (c != a):
                y = a
            else:
                pass
        fw.write("%d\n" % y)
    fw.close()


if __name__ == '__main__':
    # single(result_files[2])
    # ensemble()
    single("test_result_char_gate_word_59000.txt")
