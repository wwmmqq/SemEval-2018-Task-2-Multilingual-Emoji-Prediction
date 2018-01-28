# coding: utf-8
data_dir = '/home/wmq/Desktop/DeepText/SemEval2018Task2/data/emoji'

trial_x_ids_file = data_dir + '/raw/tweet_by_ID_us.txt.ids'
trial_x_file = data_dir + '/raw/us_trial.text'
trial_y_file = data_dir + '/raw/us_trial.labels'
trial_xy_file = data_dir + '/us_trial.txt'
source_x_file = data_dir + '/raw/tweet_by_ID_us.txt.text'
source_y_file = data_dir + '/raw/tweet_by_ID_us.txt.labels'
result_file = data_dir + '/us_train.txt'


def delete():
    trial_x = {}
    with open(trial_x_file, 'r') as frx, open(trial_y_file, 'r') as fry:
        for x, y in zip(frx, fry):
            x = x.strip()
            y = y.strip()
            trial_x[x] = y

    cnt = 0
    x_num = 0
    no_match = 0

    fw = open(result_file, 'w')
    with open(source_x_file, 'r') as frx, open(source_y_file, 'r') as fry:
        for x, y in zip(frx, fry):
            x = x.strip()
            y = y.strip()
            x_num += 1
            if x in trial_x:
                cnt += 1
                if y != trial_x[x]:
                    # print ('%d no match of x y !!' % x_num)
                    # print (x)
                    # print ('y#%s#%s#' % (y, trial_x[x]))
                    no_match += 1
            else:
                fw.write('%s\t\t%s\n' % (y, x))
    fw.close()
    print ('x_num: %d' % x_num)
    print ('repeat cnt: %d' % cnt)
    print ('train sample cnt: %d' % (x_num-cnt))
    print ('no match cnt: %d' % no_match)
    # x_num: 492496
    # repeat cnt: 21041
    # no match cnt: 1714


def merge_text_labels():
    fw = open(trial_xy_file, 'w')
    n = 0
    with open(trial_x_file, 'r') as frx, open(trial_y_file, 'r') as fry:
        for x, y in zip(frx, fry):
            x = x.strip()
            y = y.strip()
            fw.write('%s\t\t%s\n' % (y, x))
            n += 1
    print ('n : %d' % n)


def diff():
    ids = set()
    same = 0
    with open(trial_x_ids_file, 'r') as fr:
        for ID in fr:
            if ID not in ids:
                ids.add(ID)
            else:
                same += 1
    print ('samples cnt: %d' % len(ids))
    print ('same: %d' % same)


if __name__ == '__main__':
    diff()
    delete()
    merge_text_labels()
