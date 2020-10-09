import pickle
import random
import numpy as np
import csv
import ast
sep_filename = "sep_topic_vecs.txt"

with open('raw_data.pickle', 'rb') as f_data:
    data_dict = pickle.load(f_data)
npz = np.load(open('../lda2vec/topics.pyldavis.npz', 'rb'))
topic_rep = npz['doc_topic_dists']

selected_list = random.sample(list(enumerate(data_dict['text'][data_dict['testunlab_st_ind']:data_dict['testunlab_en_ind']])), data_dict['val_en_ind'])

with open(sep_filename, 'w') as opfile:
    wr = csv.writer(opfile, delimiter = '\t')
    header = ['post','label']
    wr.writerow(header)
    for entry in selected_list:
        ind,post = entry   
        row = [post,list(topic_rep[ind])]
        wr.writerow(row)
