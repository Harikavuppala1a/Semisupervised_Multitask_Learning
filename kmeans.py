import pickle
from sklearn.cluster import KMeans
import numpy as np
import h5py
from keras.utils import to_categorical
from sent_enc_embed import bert_flat_embed_post
import random
import csv

n_clusters = [5,7,10]
embed_dim = 768
s_filename = 'sent_enc_feat~bert_pre~False.h5'

with open('raw_data.pickle', 'rb') as f_data:
	data_dict = pickle.load(f_data)
	if not os.path.isfile(s_filename):
		post_feats = bert_flat_embed_posts(data_dict['text'], embed_dim)
		with h5py.File('s_filename', "w") as hfile:
			hfile.create_dataset('feats', data=post_feats)
	with h5py.File(s_filename, "r") as hf:
		post_feat_single_kmeans = np.concatenate((hf['feats'][0:data_dict['val_en_ind']], hf['feats'][data_dict['testunlab_st_ind']:data_dict['testunlab_en_ind']]),axis=0)
		post_feat_sep_kmeans = hf['feats'][data_dict['testunlab_st_ind']:data_dict['testunlab_en_ind']]

def get_predicted_ids(cluster,post_feat_kmeans,predict_feat):
	kmeans = KMeans(n_clusters=cluster).fit(post_feat_kmeans)
	predicted_ids = kmeans.predict(predict_feat)
	return predicted_ids

def get_single_kmeans(post_feat_single_kmeans):
	for cluster in n_clusters:
		filename = ("kmeans_vecs_%s.pickle" % (cluster))
		predicted_ids = get_predicted_ids(cluster,post_feat_single_kmeans,post_feat_single_kmeans)
		with open(filename, 'wb') as kmeans_data:
			pickle.dump(predicted_ids, kmeans_data)
			
def get_sep_kmeans(post_feat_sep_kmeans,data_dict):
	selected_list = random.sample(list(enumerate(data_dict['text'][data_dict['testunlab_st_ind']:data_dict['testunlab_en_ind']])), data_dict['val_en_ind'])
	selected_list_rep = []
	for tup in selected_list:
		ind,text = tup
		selected_list_rep.append(post_feat_sep_kmeans[ind])
	for cluster in n_clusters:
		filename = ("sep_kmeans_vecs_%s.txt" % (cluster))
		predicted_ids = get_predicted_ids(cluster,post_feat_sep_kmeans,selected_list_rep)
		with open(filename, 'w') as opfile:
			wr = csv.writer(opfile, delimiter = '\t')
			header = ['post','label']
			wr.writerow(header)
			for num,entry in enumerate(selected_list):
				ind, post = entry	
				row = [post,predicted_ids[num]]
				wr.writerow(row)

get_single_kmeans(post_feat_single_kmeans)
get_sep_kmeans(post_feat_sep_kmeans,data_dict)



