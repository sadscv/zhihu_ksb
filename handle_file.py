import load as ld
import numpy as np
import csv

CHAR_EMBEDDING_FILE = './data/char_embedding.txt'
WORD_EMBEDDING_FILE = './data/word_embedding.txt'
TRAIN_DATA_FILE = './data/question_train_set.txt'
TEST_DATA_FILE = './data/question_eval_set.txt'
TOPIC_INFO_FILE = './data/topic_info.txt'
TRAIN_LABEL_FILE = './data/question_topic_train_set.txt'

char_embedding=ld.load_char_embedding()
word_embedding=ld.load_word_embedding()

topic_des=ld.load_topic_des()
question_des=ld.load_question_des()


num=0


def get_info_by_qid(question_des, question_id):
    info = question_des.loc[question_des['question_id'] == question_id]

    ct = info[['question_name_char']].values[0][0].split(",")
    wt = info[['question_name_word']].values[0][0].split(",")
    cd = info[['question_des_char']].values[0][0].split(",")
    wd = info[['question_des_word']].values[0][0].split(",")
    return ct, wt, cd, wd

with open("./data/question_topic_train_set.txt") as f:
    csvfile=open("./data/train.csv","w")
    writer=csv.writer(csvfile)
    writer.writerow(["question_id","topic_num","question_ct","question_wt","question_cd","question_wd"])
    question_topic =True
    while (question_topic):
        question_topic = f.readline().strip("\n").split("\t")
        num+=1
        question_id = question_topic[0]
        print(num,question_id)
        topic_id = question_topic[1].split(",")

        #找到question_id 对应的一行数据，并分别存入ct,wt,cd,wd中.
        #ct,wt,cd,wd 都是一个list.　
        question_ct, question_wt, question_cd, question_wd = get_info_by_qid(question_des, int(question_id))

        question_ct_embedding = []
        question_wt_embedding = []
        question_cd_embedding = []
        question_wd_embedding = []


        if (len(question_ct) ==0 or len(question_wt)==0 or len(question_cd)==0 or len(question_cd)==0):
            continue


        for question_ct_ in question_ct:
            header_list = []
            for i in range(1, 257, 1):
                header_list.append(str(i))
            if (len(char_embedding.loc[char_embedding['char'] == question_ct_]) == 0):
                temp = np.random.rand(256)
                question_ct_embedding.append(temp)
            else:
                question_ct_embedding.append(
                    char_embedding.loc[char_embedding['char'] == question_ct_][header_list].values[0])
        question_ct_embedding = np.mat(question_ct_embedding, dtype=np.float32)


        for question_wt_ in question_wt:
            header_list = []
            for i in range(1, 257, 1):
                header_list.append(str(i))
            if (len(word_embedding.loc[word_embedding['word'] == question_wt_]) == 0):
                temp = np.random.rand(256)
                question_wt_embedding.append(temp)
            else:
                question_wt_embedding.append(
                    word_embedding.loc[word_embedding['word'] == question_wt_][header_list].values[0])
        question_wt_embedding = np.mat(question_wt_embedding, dtype=np.float32)


        for question_cd_ in question_cd:
            header_list = []
            for i in range(1, 257, 1):
                header_list.append(str(i))
            if (len(char_embedding.loc[char_embedding['char'] == question_cd_]) == 0):
                temp = np.random.rand(256)
                question_cd_embedding.append(temp)
            else:
                question_cd_embedding.append(
                    char_embedding.loc[char_embedding['char'] == question_cd_][header_list].values[0])
        question_cd_embedding = np.mat(question_cd_embedding, dtype=np.float32)


        for question_wd_ in question_wd:
            header_list = []
            for i in range(1, 257, 1):
                header_list.append(str(i))
            if (len(word_embedding.loc[word_embedding['word'] == question_wd_]) == 0):
                temp = np.random.rand(256)
                question_wd_embedding.append(temp)
            else:
                question_wd_embedding.append(
                    word_embedding.loc[word_embedding['word'] == question_wd_][header_list].values[0])
        question_wd_embedding = np.mat(question_wd_embedding, dtype=np.float32)



        for topic_ in topic_id:
            index = 0
            # y_temp=np.zeros([2000])
            while (True):
                if (topic_des['topic_id'][index] == int(topic_)):
                    # y_temp[index] = 1
                    break
                index += 1
            writer.writerow([question_id,index,question_ct_embedding,question_wt_embedding,question_cd_embedding,question_wd_embedding])



    csvfile.close()