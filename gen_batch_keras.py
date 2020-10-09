import numpy as np
import keras
import random

class TrainGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, word_feats, sent_feats, data_dict, batch_size,conf_scores_array,use_conf_scores,multi_task_tl,aux_only,single_inp_tasks_list,sep_inp_tasks_list):
        self.word_feats = word_feats
        self.sent_feats = sent_feats
        self.data_dict = data_dict
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.conf_scores_array = conf_scores_array
        self.use_conf_scores = use_conf_scores
        self.multi_task_tl = multi_task_tl
        self.aux_only = aux_only
        self.single_inp_tasks_list = single_inp_tasks_list
        self.sep_inp_tasks_list = sep_inp_tasks_list

        for sep_task_tup in self.sep_inp_tasks_list:
            sep_task_name, sep_task_dict = sep_task_tup
            sep_task_dict['all_list_IDs'] = np.arange(0, sep_task_dict['train_en_ind'])
            sep_task_dict['list_ids'] = self.random_sample(sep_task_dict['all_list_IDs'])

    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        list_IDs_temp = self.list_IDs[index*self.batch_size:min(len(self.list_IDs),(index+1)*self.batch_size)]
        for sep_task_tup in self.sep_inp_tasks_list:
            sep_task_name, sep_task_dict = sep_task_tup
            sep_task_dict['list_IDs_temp'] = sep_task_dict['list_ids'][index*self.batch_size:min(len(sep_task_dict['list_ids']),(index+1)*self.batch_size)]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def random_sample(self,aux_all_IDs):
        len_list_IDs = len(self.list_IDs)
        round_list_ids = round(len_list_IDs/2) 
        round_sd_all_ids = round(len(aux_all_IDs)/2)
        sd_pos = random.sample(list(aux_all_IDs[:round_sd_all_ids]), round_list_ids)
        sd_neg = random.sample(list(aux_all_IDs[round_sd_all_ids:]), (len_list_IDs - round_list_ids))
        t = sd_pos + sd_neg
        np.random.shuffle(t)
        return t

    def on_epoch_end(self):
        np.random.shuffle(self.list_IDs)
        for sep_task_tup in self.sep_inp_tasks_list:
               sep_task_name, sep_task_dict = sep_task_tup
               sep_task_dict['list_ids'] = self.random_sample(sep_task_dict['all_list_IDs'])

    def __data_generation(self, list_IDs_temp):
        if self.aux_only == False or self.single_inp_tasks_list:       
            X_inputs = []
            for word_feat in self.word_feats:
                X_inputs.append(np.empty((len(list_IDs_temp), *word_feat['dim_shape'])))

            for i, ID in enumerate(list_IDs_temp):
                input_ind = 0    
                for word_feat in self.word_feats:
                    if 'func' in word_feat:
                        X_inputs[input_ind][i,] = word_feat['func'](ID, word_feat, self.data_dict, word_feat['filepath'] + str(ID) + '.npy', word_feat['emb_size'])
                    else:
                        X_inputs[input_ind][i,] = np.load(word_feat['filepath'] + str(ID) + '.npy')
                    input_ind += 1

            for sent_feat in self.sent_feats:
                X_inputs.append(sent_feat['feats'][list_IDs_temp])
        
        sep_X_inputs = []
        for sep_task_tup in self.sep_inp_tasks_list:
            sep_task_name, sep_task_dict = sep_task_tup
            sep_inputs =[]
            for aux_word_feat in sep_task_dict['word_feats']:
                sep_inputs.append(np.empty((len(sep_task_dict['list_IDs_temp']), *aux_word_feat['dim_shape'])))

            for i, ID in enumerate(sep_task_dict['list_IDs_temp']):
                input_ind = 0    
                for aux_word_feat in sep_task_dict['word_feats']:
                    if 'func' in aux_word_feat:
                        sep_inputs[input_ind][i,] = aux_word_feat['func'](ID, aux_word_feat, sep_task_dict, aux_word_feat['filepath'] + str(ID) + '.npy', aux_word_feat['emb_size'])
                    else:
                        sep_inputs[input_ind][i,] = np.load(aux_word_feat['filepath'] + str(ID) + '.npy')
                    input_ind += 1

            for aux_sent_feat in sep_task_dict['sent_enc_feats']:
                sep_inputs.append(aux_sent_feat['feats'][sep_task_dict['list_IDs_temp']])
            sep_X_inputs.extend(sep_inputs)

        if self.use_conf_scores:
            X_inputs.append(self.conf_scores_array[list_IDs_temp])

        labels_list = [self.labels[list_IDs_temp]]
        for single_task_tup in self.single_inp_tasks_list:
            single_task_name, single_task_dict = single_task_tup
            if single_task_name == "topic":
                labels_list.append(single_task_dict['topic_vecs'][list_IDs_temp])
            elif single_task_name == "kmeans":
                labels_list.append(single_task_dict['kmeans_vectors'][list_IDs_temp])
        for sep_task_tup in self.sep_inp_tasks_list:
            sep_task_name, sep_task_dict = sep_task_tup
            labels_list.append(sep_task_dict['trainY_list'][0][sep_task_dict['list_IDs_temp']])
                
        if self.multi_task_tl == "multi_task":
            if self.sep_inp_tasks_list:
                return X_inputs+sep_X_inputs, labels_list
            else:
                return X_inputs,labels_list

        if self.aux_only:
            if self.sep_inp_tasks_list:
                return sep_X_inputs, labels_list[1]
            else:
                return X_inputs,labels_list[1]


        return X_inputs, self.labels[list_IDs_temp]

class TestGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, word_feats, sent_feats, data_dict, batch_size,conf_scores_array,use_conf_scores):
        self.word_feats = word_feats
        self.sent_feats = sent_feats
        self.data_dict = data_dict

        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.conf_scores_array = conf_scores_array
        self.use_conf_scores = use_conf_scores

    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        list_IDs_temp = self.list_IDs[index*self.batch_size:min(len(self.list_IDs),(index+1)*self.batch_size)]

        X = self.__data_generation(list_IDs_temp)
        return X

    def __data_generation(self, list_IDs_temp):
        X_inputs = []
        for word_feat in self.word_feats:
            X_inputs.append(np.empty((len(list_IDs_temp), *word_feat['dim_shape'])))

        for i, ID in enumerate(list_IDs_temp):
            input_ind = 0       
            for word_feat in self.word_feats:
                if 'func' in word_feat:
                    X_inputs[input_ind][i,] = word_feat['func'](ID, word_feat, self.data_dict, word_feat['filepath'] + str(ID) + '.npy', word_feat['emb_size'])
                else:
                    X_inputs[input_ind][i,] = np.load(word_feat['filepath'] + str(ID) + '.npy')
                input_ind += 1

        for sent_feat in self.sent_feats:
            X_inputs.append(sent_feat['feats'][list_IDs_temp])

        if self.use_conf_scores:
            X_inputs.append(self.conf_scores_array[list_IDs_temp])

        return X_inputs