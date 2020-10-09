import os
import shutil
import h5py
import numpy as np
import pickle
from keras.utils import to_categorical

def get_elmo_embeddings(path_rearr_src,path_rearr_dst, index_filename,len_augmented):
	os.makedirs(path_rearr_dst, exist_ok=True)
	file_index = 0
	f_res = open(index_filename, 'r')
	for line in f_res.readlines():
		if file_index < len_augmented:
			row = line.strip()
			src = str(row) + ".npy"
			dst = str(file_index) + ".npy"
			shutil.copy(path_rearr_src + src,path_rearr_dst+dst)
			file_index = file_index + 1
		else:
			break

def get_bert_embeddings(path_rearr_src,path_rearr_dst, index_filename,data_dict):
	f_res = open(index_filename, 'r')
	indexes =[]
	file_index = 0
	for line in f_res.readlines():
		if file_index < data_dict['test_en_ind']:
			indexes.append(int(line.strip()))
			file_index = file_index + 1
		else:
			break
	with h5py.File(path_rearr_src,"r") as hf:
		bert_feat = hf['feats'][:data_dict['testunlab_en_ind']]
	bert_edited = [bert_feat[i] for i in indexes]
	with h5py.File(path_rearr_dst, "w") as hfile:
		hfile.create_dataset('feats', data=bert_edited) 

def get_vecs(augment_data_filename,supervised_data_filename,total_raw_data_filename,index_filename,conf_dict_com,data_dict,single_task_dict,train_ratio):
	if conf_dict_com ['augment_data']:
		aux_data_filename_aug = ("%sauxillary_main~%s~%s~%s~%s.pickle" % (conf_dict_com["save_folder_name"], conf_dict_com ['augment_data'], single_task_dict['filename'][:-7],augment_data_filename[:-7],train_ratio))
		if os.path.isfile(aux_data_filename_aug):
			with open(aux_data_filename_aug, 'rb') as vecs_data:
				data_vecs_rep = pickle.load(vecs_data)
		else:
			with open(conf_dict_com["save_folder_name"] + single_task_dict['filename'], 'rb') as vecs_data:
				vecs_data_raw = pickle.load(vecs_data)
			with open(conf_dict_com["save_folder_name"] + total_raw_data_filename, 'rb') as raw_data:
				raw_data_dict = pickle.load(raw_data)

			if len(vecs_data_raw.shape) > 1 :
				vecs_rep_test = np.zeros([len(raw_data_dict['text'][raw_data_dict['test_st_ind']:raw_data_dict['test_en_ind']]),vecs_data_raw.shape[1]])
				vecs_data_raw = np.insert(vecs_data_raw,raw_data_dict['val_en_ind'],vecs_rep_test,axis=0)
				data_vecs_rep = np.zeros([data_dict['val_en_ind'], vecs_data_raw.shape[1]])
			else: 
				vecs_rep_test = np.zeros([len(raw_data_dict['text'][raw_data_dict['test_st_ind']:raw_data_dict['test_en_ind']])])
				vecs_data_raw = np.insert(vecs_data_raw,raw_data_dict['val_en_ind'],vecs_rep_test)
				data_vecs_rep = np.zeros([data_dict['val_en_ind']])
			f_res = open(index_filename, 'r')
			file_index = 0
			
			for line in f_res.readlines():
				if file_index < data_dict['val_en_ind']:
					data_vecs_rep[file_index] = vecs_data_raw[int(line.strip())]
					file_index = file_index + 1
				else:
					break
			with open(aux_data_filename_aug, 'wb') as f_data:
				pickle.dump(data_vecs_rep, f_data)
	else:
		aux_data_filename_sup = ("%sauxillary_main~%s~%s~%s~%s.pickle" % (conf_dict_com["save_folder_name"], conf_dict_com ['augment_data'], single_task_dict['filename'][:-7],supervised_data_filename[:-7], train_ratio))
		if os.path.isfile(aux_data_filename_sup):
			with open(aux_data_filename_sup, 'rb') as vecs_data:
				data_vecs_rep = pickle.load(vecs_data)
		else:
			with open(conf_dict_com["save_folder_name"] + single_task_dict['filename'], 'rb') as vecs_data:
				vecs_data_raw = pickle.load(vecs_data)
			data_vecs_rep = vecs_data_raw[:data_dict['train_en_ind']]
			with open(aux_data_filename_sup, 'wb') as f_data:
				pickle.dump(data_vecs_rep, f_data)

	if single_task_dict['filename'].startswith("kmeans"):
		if conf_dict_com ['augment_data']:
			kmeans_categ_data_filename = ("%sauxillary_kmeans_categ_data~%s~%s~%s~%s.pickle" % (conf_dict_com["save_folder_name"], conf_dict_com ['augment_data'], single_task_dict['filename'][:-7],augment_data_filename[:-7],train_ratio))
		else:
			kmeans_categ_data_filename = ("%sauxillary_kmeans_categ_data~%s~%s~%s~%s.pickle" % (conf_dict_com["save_folder_name"], conf_dict_com ['augment_data'], single_task_dict['filename'][:-7],supervised_data_filename[:-7],train_ratio))
		if os.path.isfile(kmeans_categ_data_filename):
			with open(kmeans_categ_data_filename, 'rb') as d:
				kmeans_categ_data = pickle.load(d)
		else:
			kmeans_categ_data = to_categorical(data_vecs_rep, single_task_dict['n_clusters'])
			with open(kmeans_categ_data_filename, 'wb') as d:
				pickle.dump(kmeans_categ_data, d)
		single_task_dict["kmeans_ids"] = data_vecs_rep
		single_task_dict["kmeans_vectors"] = kmeans_categ_data
	else:
		single_task_dict["topic_vecs"] = data_vecs_rep
