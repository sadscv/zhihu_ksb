import numpy as np


def get_char_embeddings():
    c2v = dict()
    with open("data/char_embedding.txt") as f:
        first = f.readline()
        cnum, d = first.rstrip(" \r\n").split()
        print("- Reading {} char embeddings of dim {}".format(cnum, d))
        for line in f:
            strs = line.rstrip(" \r\n").split()
            k = strs[0]
            v = [float(x) for x in strs[1:]]
            c2v[k] = np.asarray(v)
        print("- Done.")
    return c2v


def get_word_embeddings():
    w2v = dict()
    with open("data/word_embedding.txt") as f:
        first = f.readline()
        cnum, d = first.rstrip(" \r\n").split()
        print("- Reading {} word embeddings of dim {}".format(cnum, d))
        num = 0
        for line in f:
            if num % 20000 == 0:
                print('{} lines readed'.format(num))
            num += 1
            strs = line.rstrip(" \r\n").split()
            k = strs[0]
            v = [float(x) for x in strs[1:]]
            w2v[k] = np.asarray(v)
        print("- Done.")
    return w2v


def get_topic_info():
    fathers = dict()
    topic_name_chars = dict()
    topic_name_words = dict()
    topic_desc_chars = dict()
    topic_desc_words = dict()
    with open("data/topic_info.txt") as f:
        for line in f:
            strs = line.rstrip(" \r\n").split()
            t = strs[0]
            fathers[t] = strs[1].split(",")
            topic_name_chars[t] = strs[2].split(',')
            topic_name_words[t] = strs[3].split(',')
            topic_desc_chars[t] = strs[4].split(',')
            topic_desc_words[t] = strs[5].split(',')
    return fathers, topic_name_chars, topic_name_words, topic_desc_chars, topic_desc_words

if __name__ == "__main__":
    c2v = get_char_embeddings()
    w2v = get_word_embeddings()
    pass