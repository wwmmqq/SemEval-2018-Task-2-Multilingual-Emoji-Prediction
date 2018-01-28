# coding: utf-8

# 0	❤	_red_heart_
# 1	😍	_smiling_face_with_hearteyes_
# 2	😂	_face_with_tears_of_joy_
# 3	💕	_two_hearts_
# 4	🔥	_fire_
# 5	😊	_smiling_face_with_smiling_eyes_
# 6	😎	_smiling_face_with_sunglasses_
# 7	✨	_sparkles_
# 8	💙	_blue_heart_
# 9	😘	_face_blowing_a_kiss_
# 10	📷	_camera_
# 11	🇺🇸	_United_States_
# 12	☀	_sun_
# 13	💜	_purple_heart_
# 14	😉	_winking_face_
# 15	💯	_hundred_points_
# 16	😁	_beaming_face_with_smiling_eyes_
# 17	🎄	_Christmas_tree_
# 18	📸	_camera_with_flash_
# 19	😜	_winking_face_with_tongue_

import numpy as np
from gensim.models.keyedvectors import KeyedVectors

mapping = {
    '❤': '0', '😍': '1', '😂': '2', '💕': '3', '🔥': '4', '😊': '5', '😎': '6', '✨': '7', '💙': '8',
    '😘': '9', '📷': '10', '🇺🇸': '11', '☀': '12', '💜': '13', '😉': '14', '💯': '15', '😁': '16',
    '🎄': '17', '📸': '18', '😜': '19'}
emoji_list = [x for x in mapping.keys()]
idx2emoji = {i: str.decode(e, 'utf-8') for i, e in enumerate(emoji_list)}
emoji2idx = {str.decode('😂', 'utf-8'): i for i, e in enumerate(emoji_list)}


def get_emoji():
    model = KeyedVectors.load_word2vec_format(
        '/home/wmq/Desktop/DeepText/SemEval2018Task2/data/embed/model_swm_300-6-10-low.w2v',
        binary=False)

    emoji_we = []
    for i in range(20):
        emoji_we.append(model[idx2emoji[i]])

    with open("emoji_we.txt", 'w') as fw:
        for idx, w in enumerate(emoji_we):
            t = " ".join(["%0.6f" % x for x in w])
            fw.write("%s %s\n" % (idx, t))


def load_emoji(in_file, transpose=True):
    emoji_vector = []
    with open(in_file, 'r') as fr:
        for line in fr:
            emoji_vector.append([float(t) for t in line.split()[1:]])
    emoji_embed = np.array(emoji_vector, dtype=np.float32)
    if transpose:
        emoji_embed = np.transpose(emoji_embed)
    print("load emoji embedding: (%d, %d)" % emoji_embed.shape)
    return emoji_embed
