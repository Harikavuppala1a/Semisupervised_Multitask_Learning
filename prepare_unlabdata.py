import os
import numpy as np
import time
from dlModels import get_model, attLayer_hier, multi_binary_loss, br_binary_loss, lp_categ_loss, multi_cat_w_loss, multi_cat_loss, kmax_pooling
from sklearn.utils import class_weight
from loadPreProc import *
from evalMeasures import *
import sys
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sent_enc_embed import  *
from word_embed import *
from keras import backend as K
from gen_batch_keras import TrainGenerator, TestGenerator
from neuralApproaches import class_imb_loss_nonlin,transform_labels
import pickle
import json
import csv 
import h5py
import shutil
import re
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from random import sample 
import math

sys.setrecursionlimit(10000)
conf_dict_list, conf_dict_com = load_config(sys.argv[1])
print (conf_dict_com)
os.environ["CUDA_VISIBLE_DEVICES"] = conf_dict_com['GPU_ID']
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

def load_data_new(filename, data_path, save_path, test_ratio, valid_ratio, rand_state, max_words_sent, test_mode):
  data_dict_filename = ("%sraw_data~%s~%s~%s~%s~%s~%s~%s.pickle" % (save_path, filename[:-4], test_ratio, valid_ratio, rand_state, max_words_sent, test_mode, conf_dict_com['st_variant'].startswith('hard')))
  cl_in_filename = ("%sraw_data~%s~%s.pickle" % (save_path, filename[:-4], max_words_sent))
  

  if os.path.isfile(data_dict_filename):
    print("loading input data")
    with open(data_dict_filename, 'rb') as f_data:
        data_dict = pickle.load(f_data)       
  else:      
    if os.path.isfile(cl_in_filename):
      print("loading cleaned unshuffled input")
      with open(cl_in_filename, 'rb') as f_cl_in:
          text, text_sen, label_lists, text_unlab, text_unlab_sen = pickle.load(f_cl_in)
    else:
      r_anum = re.compile(r'([^\sa-z0-9.(?)!])+')
      r_white = re.compile(r'[\s.(?)!]+')
      text = []; label_lists = []; text_sen = []; text_unlab = []; text_unlab_sen = []
      with open(data_path + 'unlab_minus_lab_shortest_n.txt', 'r') as unlabfile:
        reader_unlab = unlabfile.readlines()
        for row_unlab in reader_unlab:
            post_unlab = str(row_unlab)
            row_clean = r_white.sub(' ', r_anum.sub('', post_unlab.lower())).strip()
            text_unlab.append(row_clean)
            se_list_unlab = []
            for se in sent_tokenize(post_unlab):
                se_clean = r_white.sub(' ', r_anum.sub('', str(se).lower())).strip()
                if se_clean == "":
                    continue
                words_unlab = se_clean.split(' ')
                while len(words_unlab) > max_words_sent:
                    se_list_unlab.append(' '.join(words_unlab[:max_words_sent]))
                    words_unlab = words_unlab[max_words_sent:]
                se_list_unlab.append(' '.join(words_unlab))
            text_unlab_sen.append(se_list_unlab)

      with open(data_path + filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter = '\t')
        for row in reader:
          post = str(row['post'])
          row_clean = r_white.sub(' ', r_anum.sub('', post.lower())).strip()
          text.append(row_clean)

          se_list = []
          for se in sent_tokenize(post):
            se_cl = r_white.sub(' ', r_anum.sub('', str(se).lower())).strip()
            if se_cl == "":
                continue
            words = se_cl.split(' ')
            while len(words) > max_words_sent:
              se_list.append(' '.join(words[:max_words_sent]))
              words = words[max_words_sent:]
            se_list.append(' '.join(words))
          text_sen.append(se_list)
          cat_list = str(row['labels']).split(',')
          label_ids = list(set([LABEL_MAP[cat] for cat in cat_list]))
          label_lists.append(label_ids)

      print("saving cleaned unshuffled input")
      with open(cl_in_filename, 'wb') as f_cl_in:
        pickle.dump([text, text_sen, label_lists, text_unlab, text_unlab_sen], f_cl_in)
 
    data_dict = {}  
    data_dict['text'], data_dict['text_sen'], data_dict['lab'] = shuffle(text, text_sen, label_lists, random_state = rand_state)
    
    #concatenating label data with unlabel
    data_dict['tot_data'] = data_dict['text'] + text_unlab
    tot_data_sen = data_dict['text_sen'] + text_unlab_sen
    data_dict['text'] = data_dict['tot_data']
    data_dict['text_sen'] = tot_data_sen

    train_index = int((1 - test_ratio - valid_ratio)*len(text)+0.5)
    val_index = int((1 - test_ratio)*len(text)+0.5)

    data_dict['max_num_sent'] = max([len(post_sen) for post_sen in data_dict['text_sen'][:val_index]])
    data_dict['max_post_length'] = max([len(post.split(' ')) for post in data_dict['text'][:val_index]])
    data_dict['max_words_sent'] = max_words_sent

    assert(test_mode == False)
    if test_mode == False and conf_dict_com['st_variant'].startswith('hard'):

        data_dict['train_en_ind'] =round(train_index*0.8)
        data_dict['held_train_st_ind'] = data_dict['train_en_ind']
        data_dict['held_train_en_ind'] = train_index
        
    else:
        data_dict['train_en_ind'] = train_index
    data_dict['val_st_ind'] = train_index
    data_dict['val_en_ind'] = val_index
    data_dict['test_st_ind'] = val_index
    data_dict['test_en_ind'] = len(text)
    data_dict['testunlab_st_ind'] = len(text)
    data_dict['testunlab_en_ind'] = len(data_dict['tot_data'])
    
    print("saving input data")
    with open(data_dict_filename, 'wb') as f_data:
        pickle.dump(data_dict, f_data)

  return data_dict,data_dict_filename,cl_in_filename

def gen_postvecs(fname_mod,test_st,test_en,word_feats, sent_enc_feats, data_dict, model_type, num_cnn_filters, rnn_type, fname_part, loss_func, cw, nonlin, out_vec_size, rnn_dim, att_dim, max_pool_k_val,stack_rnn_flag,cnn_kernel_set, save_folder_name, use_saved_model, gen_att, learn_rate, dropO1, dropO2,b_size, epochs, save_model,st_variant):
	model, att_mod , post_vec_mod = get_model(model_type, data_dict['max_post_length'], data_dict['max_num_sent'], data_dict['max_words_sent'], word_feats, sent_enc_feats, learn_rate, dropO1, dropO2, num_cnn_filters, rnn_type, loss_func, nonlin, out_vec_size, rnn_dim, att_dim, max_pool_k_val, stack_rnn_flag, cnn_kernel_set, None, st_variant)

	model.load_weights(fname_mod)
	test_generator = TestGenerator(np.arange(test_st,test_en), word_feats, sent_enc_feats, data_dict, b_size)
	post_vec_op = model.predict_generator(generator=test_generator, verbose=1, use_multiprocessing=False, workers=1)
	return post_vec_op

def train_predict_new(phase,train_en,test_st,test_en,word_feats, sent_enc_feats, trainY, data_dict, model_type, num_cnn_filters, rnn_type, fname_part, loss_func, class_w, nonlin, out_vec_size, rnn_dim, att_dim, max_pool_k_val, stack_rnn_flag, cnn_kernel_set, m_ind, run_ind, save_folder_name, use_saved_model, gen_att, learn_rate, dropO1, dropO2, batch_size, num_epochs, save_model, bert_max_seq_length, st_variant,conf_scores_array,use_conf_scores):
    att_op = None
    fname_mod_op = ("%s%s/iop~%d~%d~%s.pickle" % (save_folder_name, fname_part, m_ind, run_ind, phase))
    fname_mod = ("%s%s/mod~%d~%d.h5" % (save_folder_name, fname_part, m_ind, run_ind))
    if use_saved_model  and os.path.isfile(fname_mod_op):
        print("loading model o/p")
        with open(fname_mod_op, 'rb') as f:
            mod_op = pickle.load(f)
    else:
        if use_saved_model and os.path.isfile(fname_mod):
            print("loading model")
            model, model_multitask_tr, att_mod , post_vec = get_model(model_type, data_dict['max_post_length'], data_dict['max_num_sent'], data_dict['max_words_sent'], word_feats, sent_enc_feats, learn_rate, dropO1, dropO2, num_cnn_filters, rnn_type, loss_func, nonlin, out_vec_size, rnn_dim, att_dim, max_pool_k_val, stack_rnn_flag, cnn_kernel_set, st_variant,use_conf_scores,None,None,[],[])
            model.load_weights(fname_mod)
        else:
            model, multitask_tr,att_mod , post_vec = get_model(model_type, data_dict['max_post_length'], data_dict['max_num_sent'], data_dict['max_words_sent'], word_feats, sent_enc_feats, learn_rate, dropO1, dropO2, num_cnn_filters, rnn_type, loss_func, nonlin, out_vec_size, rnn_dim, att_dim, max_pool_k_val, stack_rnn_flag, cnn_kernel_set, st_variant,use_conf_scores,None,None,[],[])
            training_generator = TrainGenerator(np.arange(0, train_en), trainY, word_feats, sent_enc_feats, data_dict, batch_size,conf_scores_array,use_conf_scores,None,False,[],[])
            model.fit_generator(generator=training_generator, epochs=num_epochs, shuffle=False, verbose=1, use_multiprocessing=False, workers=1)
            if save_model:    
                print("saving model")
                os.makedirs(save_folder_name + fname_part, exist_ok=True)
                model.save(fname_mod)
        test_generator = TestGenerator(np.arange(test_st,test_en), word_feats, sent_enc_feats, data_dict, batch_size,conf_scores_array,use_conf_scores)
        mod_op = model.predict_generator(generator=test_generator, verbose=1, use_multiprocessing=False, workers=1)
        
        if save_model:    
            print("saving model o/p")
            os.makedirs(save_folder_name + fname_part, exist_ok=True)
            with open(fname_mod_op, 'wb') as f:
                pickle.dump(mod_op, f)
            if run_ind == 0 and m_ind == 0:
                with open("%s%s/mod_sum.txt" % (save_folder_name, fname_part),'w') as fh:
                    model.summary(print_fn=lambda x: fh.write(x + '\n'))                                                                
        K.clear_session()
    return mod_op, att_op

def insight_iter(new_lab_list,prob_predicted,iter,output_folder_name):
    pos_prob = []
    neg_prob = []
    tsv_path = output_folder_name + '/' + 'insight'  + str(iter) + '.txt'
    if os.path.isfile(tsv_path):
        f_tsv = open(tsv_path, 'a')
    else:
        f_tsv = open(tsv_path, 'w')
        f_tsv.write("pred_noof_samples\tavg_pos_prb\tavg_neg_prb\n") 
    for i in range(len(new_lab_list)):
        count_labels = list(new_lab_list[i]).count(1)
        for j in range(NUM_CLASSES):
            if new_lab_list[i][j] == 1:
                pos_prob.append(prob_predicted[i][j])
            else:
                neg_prob.append(prob_predicted[i][j])
        f_tsv.write("%d\t%.3f\t%.3f\n"  % (count_labels,np.mean(pos_prob),np.mean(neg_prob)))

def get_conf_scores(mod_probs):
    conf_scores_persample = np.zeros(NUM_CLASSES)
    for prob_ind, prob in enumerate(mod_probs):           
        if prob >=0.5:
            conf_scores_persample[prob_ind] = prob
        else:
            conf_scores_persample[prob_ind] = 1-prob
    return conf_scores_persample

def generate_unlab_diversity_favour_intersection(mod_op_list, data_dict, bac_map, prob_trans_type,iter,save_path,conf_dict_com,post_rep_filename):
    new_data_list_div_fav = []
    new_data_list_sen_div_fav = []
    intersect_labels = []
    intersect_data_pos  = []
    intersect_confscore = []
    intersect_original_probs=[]

    original_st_variant = conf_dict_com['st_variant']
    split_variants = conf_dict_com['st_variant'].split('_')
    conf_dict_com['st_variant'] = split_variants[1]
    *_,new_lab_div,data_pos_div, predicted_ex_div,conf_scores_div,original_probs_div = generate_unlab_diversity(mod_op_list, data_dict, bac_map, prob_trans_type,iter,save_path,conf_dict_com,post_rep_filename)
    print (len(data_pos_div))
    conf_dict_com['st_variant'] = split_variants[2]
    *_,new_lab_fav,data_pos_fav, predicted_ex_fav,conf_scores_fav,original_probs_fav = generate_unlab_favour(mod_op_list, data_dict, bac_map, prob_trans_type,iter,save_path, conf_dict_com)
    print (len(data_pos_fav))
    intersect_div_fav_pos = [(ind,selected_pos) for ind, selected_pos in enumerate(data_pos_div) if selected_pos in data_pos_fav]
    print (len(intersect_div_fav_pos))

    for val in intersect_div_fav_pos:
        ind, selected_pos = val
        new_data_list_div_fav.append(data_dict['text'][selected_pos])
        new_data_list_sen_div_fav.append(data_dict['text_sen'][selected_pos])
        intersect_labels.append(new_lab_div[ind])
        if conf_dict_com['gen_conf_scores']:         
            intersect_confscore.append(conf_scores_div[ind])
            intersect_original_probs.append(original_probs_div[ind])
        intersect_data_pos.append(selected_pos)
    conf_dict_com['st_variant'] = original_st_variant
    return new_data_list_div_fav , new_data_list_sen_div_fav, intersect_labels,intersect_data_pos,len(intersect_labels),intersect_confscore,intersect_original_probs


def generate_unlab_diversity_favour(mod_op_list, data_dict, bac_map, prob_trans_type,iter,save_path,conf_dict_com,post_rep_filename):
    count_predicted = 0
    avg_dissimilarity = {}
    data_pos =[]
    new_data_list_diverse_div_fav = []
    new_data_list_sen_diverse_div_fav = []
    position_diverse = []
    new_lab_list_diverse_div_fav = []
    confidence_scores = []
    original_pred_probs =[]

    favour_weak = dict.fromkeys(range(NUM_CLASSES),0 )
    true_vals = data_dict['lab'][:data_dict['train_en_ind']]
    trainY_list = trans_labels_multi_hot(true_vals)

    for labels in trainY_list:
        for i in range(len(labels)):
            if labels[i] == 1:
                if i in favour_weak:
                    favour_weak[i] += 1

    with open(post_rep_filename, 'rb') as f:
        post_vec = pickle.load(f) 

    position = data_dict['testunlab_st_ind']
    print (post_vec[0][1])
    for cl_ind, tup_val in enumerate(mod_op_list):
        mod_op, att_op = tup_val
        print (len(post_vec))
        for i,pb in enumerate(mod_op):
            y_pred = np.rint(pb).astype(int)
            if sum(y_pred) == 0:
                y_pred[np.argmax(pb)] = 1
            count_labels = list(y_pred).count(1)
            if count_labels >=conf_dict_com['min_num_pred_labs'] and count_labels <=conf_dict_com['max_num_pred_labs']:
                category_coun = 0
                for j in pb:
                    if j >= conf_dict_com['confidence_thr'] or (1-j) >= conf_dict_com['confidence_thr']:
                        category_coun = category_coun + 1
                        if category_coun == NUM_CLASSES:
                            count_predicted  = count_predicted + 1
                            data_pos.append(position)
                            avg_dist = []
                            overall_support=[]
                            support_weak = []
                            weak_labels = False
                            if conf_dict_com['st_variant'] == 'combine_diversity.label_support.weakest':
                                for pred_lab in range(len(y_pred)):
                                    if y_pred[pred_lab] == 1:                                        
                                        dist_perlabel = []
                                        for pos in range(data_dict['train_en_ind']):
                                            if pred_lab in data_dict['lab'][pos]:
                                                dist_perlabel.append(cosine_distances(np.array([post_vec[position]]),np.array([post_vec[pos]])))
                                        avg_dist.append(np.mean(np.array(dist_perlabel)))
                                        if favour_weak[pred_lab] < conf_dict_com['weak_support_score']:
                                            weak_labels = True
                                            support_weak.append(favour_weak[pred_lab])
                                        overall_support.append(favour_weak[pred_lab])
                                if weak_labels == False:
                                    support = overall_support
                                else:
                                    support = support_weak
                                avg_dissimilarity[position] = np.mean(avg_dist)/np.mean(support)
                                                
                            elif conf_dict_com['st_variant'] == 'combine_diversity.uniform_support.uniform':

                                for pos in range(data_dict['train_en_ind']):
                                    avg_dist.append(cosine_distances(np.array([post_vec[position]]),np.array([post_vec[pos]])))
                                for pred_lab in range(len(y_pred)):
                                    if y_pred[pred_lab] == 1:
                                        overall_support.append(favour_weak[pred_lab])
                                avg_dissimilarity[position] = np.mean(avg_dist)/np.mean(overall_support)

                            elif conf_dict_com['st_variant'] == 'combine_diversity.label_support.uniform':
                                for pred_lab in range(len(y_pred)):
                                    if y_pred[pred_lab] == 1:
                                        dist_perlabel = []
                                        for pos in range(data_dict['train_en_ind']):
                                            if pred_lab in data_dict['lab'][pos]:
                                                dist_perlabel.append(cosine_distances(np.array([post_vec[position]]),np.array([post_vec[pos]])))
                                        avg_dist.append(np.mean(np.array(dist_perlabel)))
                                        overall_support.append(favour_weak[pred_lab])
                                avg_dissimilarity[position] = np.mean(avg_dist)/np.mean(overall_support)

                            elif conf_dict_com['st_variant'] == 'combine_diversity.uniform_support.weakest':
                                for pos in range(data_dict['train_en_ind']):
                                    avg_dist.append(cosine_distances(np.array([post_vec[position]]),np.array([post_vec[pos]])))
                                for pred_lab in range(len(y_pred)):
                                    if y_pred[pred_lab] == 1:
                                        if favour_weak[pred_lab] < conf_dict_com['weak_support_score']:
                                            weak_labels = True
                                            support_weak.append(favour_weak[pred_lab])
                                        overall_support.append(favour_weak[pred_lab])
                                if weak_labels == False:
                                    support = overall_support
                                else:
                                    support = support_weak
                                avg_dissimilarity[position] = np.mean(avg_dist)/np.mean(support)

                    else:
                        break
            position = position + 1  
    top_k = conf_dict_com['retaining_ratio'] * len(data_pos)
    
    ind = 1
    for key, value in sorted(avg_dissimilarity.items(),key=lambda x:x[1],reverse=True):
        if ind <= top_k:
            position_diverse.append(key)
            ind = ind + 1
        else:
            break
    position_diverse.sort()

    for pos in position_diverse:
        posi = pos - data_dict['testunlab_st_ind']
        new_data_list_diverse_div_fav.append(data_dict['text'][pos])
        new_data_list_sen_diverse_div_fav.append(data_dict['text_sen'][pos])
        if conf_dict_com['gen_conf_scores']:
            conf_scores_persample = get_conf_scores(mod_op[posi])      
            confidence_scores.append(conf_scores_persample)
            original_pred_probs.append(mod_op[posi])
        y_pred = np.rint(mod_op[posi]).astype(int)
        if sum(y_pred) == 0:
            y_pred[np.argmax(mod_op[posi])] = 1
        new_lab_list_diverse_div_fav.append(y_pred)
    pred_vals = di_op_to_label_lists(new_lab_list_diverse_div_fav)
    
    return new_data_list_diverse_div_fav, new_data_list_sen_diverse_div_fav, pred_vals,position_diverse,len(new_data_list_diverse_div_fav),confidence_scores,original_pred_probs
    
def generate_unlab_favour(mod_op_list, data_dict, bac_map, prob_trans_type, iter,save_path,conf_dict_com):
    count_predicted = 0
    data_pos =[]
    new_data_list_favour = []
    new_data_list_sen_favour = []
    pred_vals = []
    position_favour = []
    confidence_scores = []
    original_pred_probs =[]

    predict_favour_weak ={}
    favour_weak = dict.fromkeys(range(NUM_CLASSES),0 )
    true_vals = data_dict['lab'][:data_dict['train_en_ind']]
    trainY_list = trans_labels_multi_hot(true_vals)

    for labels in trainY_list:
        for i in range(len(labels)):
            if labels[i] == 1:
                if i in favour_weak:
                    favour_weak[i] += 1

    position = data_dict['testunlab_st_ind']

    for cl_ind, tup_val in enumerate(mod_op_list):
        mod_op, att_op = tup_val    
        for i,pb in enumerate(mod_op):
        	y_pred = np.rint(pb).astype(int)
        	if sum(y_pred) == 0:
        		y_pred[np.argmax(pb)] = 1
        	count_labels = list(y_pred).count(1)
        	if count_labels >=conf_dict_com['min_num_pred_labs'] and count_labels <=conf_dict_com['max_num_pred_labs']:
        		category_coun = 0
        		for j in pb:
        			if j >= conf_dict_com['confidence_thr'] or (1-j) >= conf_dict_com['confidence_thr']:
        				category_coun = category_coun + 1
        				if category_coun == NUM_CLASSES:
        					count_predicted  = count_predicted + 1
        					data_pos.append(position)
        					support_weak = []
        					overall_support = []
        					weak_labels =False
        					for i in range(len(y_pred)):
        						if y_pred[i] == 1:
        							if favour_weak[i] < conf_dict_com['weak_support_score'] and conf_dict_com['st_variant'] == 'support.weakest' :
        								weak_labels = True
        								support_weak.append(favour_weak[i])
        							overall_support.append(favour_weak[i])
        					if weak_labels == False:
        						support = overall_support
        					else:
        						support = support_weak
        					predict_favour_weak[position] = np.mean(support)
        			else:
        				break
        	position = position + 1   
    top_k = conf_dict_com['retaining_ratio'] * len(data_pos)
    ind = 1

    for key, value in sorted(predict_favour_weak.items(),key=lambda x:x[1],reverse=False):

        if ind <= top_k:
            position_favour.append(key)
            ind = ind + 1
        else:
        	break
    position_favour.sort()

    for pos in position_favour:
        posi = pos - data_dict['testunlab_st_ind']
        new_data_list_favour.append(data_dict['text'][pos])
        new_data_list_sen_favour.append(data_dict['text_sen'][pos])
        if conf_dict_com['gen_conf_scores']:
            conf_scores_persample = get_conf_scores(mod_op[posi])      
            confidence_scores.append(conf_scores_persample)
            original_pred_probs.append(mod_op[posi])
        y_pred = np.rint(mod_op[posi]).astype(int)
        if sum(y_pred) == 0:
            y_pred[np.argmax(mod_op[posi])] = 1
        pred_vals.append(y_pred)
    y_pred_vals = di_op_to_label_lists(pred_vals)

    return new_data_list_favour, new_data_list_sen_favour, y_pred_vals,position_favour,len(y_pred_vals),confidence_scores,original_pred_probs

def generate_unlab_diversity(mod_op_list, data_dict, bac_map, prob_trans_type,iter,save_path,conf_dict_com,post_rep_filename):
    count_predicted = 0
    avg_dissimilarity = {}
    data_pos =[]
    new_data_list_diverse = []
    new_data_list_sen_diverse = []
    position_diverse = []
    new_lab_list_diverse = []
    confidence_scores = []
    original_pred_probs =[]
    with open(post_rep_filename, 'rb') as f:
    	post_vec = pickle.load(f) 

    position = data_dict['testunlab_st_ind']
    for cl_ind, tup_val in enumerate(mod_op_list):
        mod_op, att_op = tup_val
        for i,pb in enumerate(mod_op):
            y_pred = np.rint(pb).astype(int)
            if sum(y_pred) == 0:
                y_pred[np.argmax(pb)] = 1
            count_labels = list(y_pred).count(1)
            if count_labels >=conf_dict_com['min_num_pred_labs'] and count_labels <=conf_dict_com['max_num_pred_labs']:
                category_coun = 0
                for j in pb:
                    if j >= conf_dict_com['confidence_thr'] or (1-j) >= conf_dict_com['confidence_thr']:
                        category_coun = category_coun + 1
                        if category_coun == NUM_CLASSES:
                            count_predicted  = count_predicted + 1
                            data_pos.append(position)
                            avg_dist = []
                            if conf_dict_com['st_variant'] == 'diversity.label':
                                for pred_lab in range(len(y_pred)):
                                    if y_pred[pred_lab] == 1:
                                        dist_perlabel = []
                                        for pos in range(data_dict['train_en_ind']):
                                            if pred_lab in data_dict['lab'][pos]:
                                                dist_perlabel.append(cosine_distances(np.array([post_vec[position]]),np.array([post_vec[pos]])))
                                        avg_dist.append(np.mean(np.array(dist_perlabel)))
                            elif conf_dict_com['st_variant'] == 'diversity.uniform':
                            	for pos in range(data_dict['train_en_ind']):
                            		avg_dist.append(cosine_distances(np.array([post_vec[position]]),np.array([post_vec[pos]])))
                            avg_dissimilarity[position] = np.mean(avg_dist)
                    else:
                        break
            position = position + 1  
    top_k = conf_dict_com['retaining_ratio'] * len(data_pos)
    
    ind = 1
    for key, value in sorted(avg_dissimilarity.items(),key=lambda x:x[1],reverse=True):
        if ind <= top_k:
            position_diverse.append(key)
            ind = ind + 1
        else:
        	break
    position_diverse.sort()
    
    for pos in position_diverse:
        posi = pos - data_dict['testunlab_st_ind']
        new_data_list_diverse.append(data_dict['text'][pos])
        new_data_list_sen_diverse.append(data_dict['text_sen'][pos])
        if conf_dict_com['gen_conf_scores']:
            conf_scores_persample = get_conf_scores(mod_op[posi])      
            confidence_scores.append(conf_scores_persample)
            original_pred_probs.append(mod_op[posi])
        y_pred = np.rint(mod_op[posi]).astype(int)
        if sum(y_pred) == 0:
            y_pred[np.argmax(mod_op[posi])] = 1
        new_lab_list_diverse.append(y_pred)
    pred_vals = di_op_to_label_lists(new_lab_list_diverse)
      
    return new_data_list_diverse, new_data_list_sen_diverse, pred_vals,position_diverse,len(new_data_list_diverse),confidence_scores,original_pred_probs


def generate_unlab_IPSPC(mod_op_list, data_dict, bac_map, prob_trans_type,iter,save_path, conf_dict_com):
    count_predicted = 0
    new_data_list = []
    new_data_list_sen =[]
    new_lab_list = []
    data_pos =[]
    prob_predicted = []
    confidence_scores = []
    original_pred_probs =[]

    position = data_dict['testunlab_st_ind']
    for cl_ind, tup_val in enumerate(mod_op_list):
        mod_op, att_op = tup_val     
        for i,pb in enumerate(mod_op):
            y_pred = np.rint(pb).astype(int)
            if sum(y_pred) == 0:
                y_pred[np.argmax(pb)] = 1
            count_labels = list(y_pred).count(1)
            if count_labels >=conf_dict_com['min_num_pred_labs'] and count_labels <=conf_dict_com['max_num_pred_labs']:
                category_coun = 0
                for j in pb:
                    if j >= conf_dict_com['confidence_thr'] or (1-j) >= conf_dict_com['confidence_thr']:
                        category_coun = category_coun + 1
                        if category_coun == NUM_CLASSES:
                            count_predicted  = count_predicted + 1
                            new_data_list.append(data_dict['text'][position])
                            new_data_list_sen.append(data_dict['text_sen'][position])
                            data_pos.append(position)
                            new_lab_list.append(y_pred)
                            prob_predicted.append(pb)
                            if conf_dict_com['gen_conf_scores']:
                                conf_scores_persample = get_conf_scores(pb)      
                                confidence_scores.append(conf_scores_persample)
                                original_pred_probs.append(pb)
                    else:
                        break
            position = position + 1 
    pred_vals = di_op_to_label_lists(new_lab_list)
    if conf_dict_com['insights_iteration']:
        insight_iter(new_lab_list,prob_predicted,iter,conf_dict_com["output_folder_name"])
    return new_data_list, new_data_list_sen, pred_vals,data_pos, count_predicted,confidence_scores,original_pred_probs

def generate_unlab_basic(mod_op_list, data_dict, bac_map, prob_trans_type,iter,save_path, conf_dict_com):
    count_predicted = 0
    new_data_list = []
    new_data_list_sen =[]
    new_lab_list = []
    data_pos =[]
    confidence_scores = []
    original_pred_probs =[]

    position = data_dict['testunlab_st_ind']
    for cl_ind, tup_val in enumerate(mod_op_list):
        mod_op, att_op = tup_val   
        for i,pb in enumerate(mod_op):
            y_pred = np.rint(pb).astype(int)
            if sum(y_pred) == 0:
                y_pred[np.argmax(pb)] = 1
            category_coun = 0
            for j in pb:
                    if j >= conf_dict_com['confidence_thr'] or (1-j) >= conf_dict_com['confidence_thr']:
                        category_coun = category_coun + 1
                        if category_coun == NUM_CLASSES:
                            count_predicted  = count_predicted + 1
                            new_data_list.append(data_dict['text'][position])
                            new_data_list_sen.append(data_dict['text_sen'][position])
                            data_pos.append(position)
                            new_lab_list.append(y_pred)
                            if conf_dict_com['gen_conf_scores']:
                                conf_scores_persample = get_conf_scores(pb)      
                                confidence_scores.append(conf_scores_persample)
                                original_pred_probs.append(pb)
                    else:
                        break
            position = position + 1 
    pred_vals = di_op_to_label_lists(new_lab_list)

    return new_data_list, new_data_list_sen, pred_vals,data_pos, count_predicted,confidence_scores,original_pred_probs


def generate_unlab_random(mod_op_list, data_dict, bac_map, prob_trans_type,iter,save_path, conf_dict_com):
    count_predicted = 0
    new_data_list = []
    new_data_list_sen =[]
    new_lab_list = []
    data_pos =[]
    confidence_scores = []
    original_pred_probs =[]

    position = data_dict['testunlab_st_ind']
    for cl_ind, tup_val in enumerate(mod_op_list):
        mod_op, att_op = tup_val
        list1 = list(range(0, 70000))
        index_rand = sample(list1, 5113)
        for ind in index_rand:
            new_data_list.append(data_dict['text'][ind+position])
            new_data_list_sen.append(data_dict['text_sen'][ind+position])
            data_pos.append(ind+position)
            y_pred = np.rint(mod_op[ind]).astype(int)
            if sum(y_pred) == 0:
                y_pred[np.argmax(mod_op[ind])] = 1
            new_lab_list.append(y_pred)
            if conf_dict_com['gen_conf_scores']:
                conf_scores_persample = get_conf_scores(mod_op[ind])      
                confidence_scores.append(conf_scores_persample)
                original_pred_probs.append(mod_op[ind])
    pred_vals = di_op_to_label_lists(new_lab_list)
    print ("predicted num of good examples",count_predicted)
    return new_data_list, new_data_list_sen, pred_vals,data_pos, count_predicted,confidence_scores,original_pred_probs

def generate_unlab_hard(heldout_op,heldout_lab,mod_op_list, data_dict, bac_map, prob_trans_type,iter,save_path,conf_dict_com,post_rep_filename):
    hardset_list= {}
    selected_hardset_perlabel_list = {k:[] for k in range(NUM_CLASSES)}
    selected_hardset_list = []
    total_prob_list = []
    print (len(heldout_op))
    print (len(heldout_lab))

    for sample_ind, prob_array in enumerate(heldout_op):
        prob_list = []
        for ind in range(len(prob_array)):
            if ind in heldout_lab[sample_ind]:
                prob_list.append(prob_array[ind])               
            else:
                prob_list.append(1-prob_array[ind])
            if prob_list[ind] < conf_dict_com['hardset_perlabel_thr']:
                selected_hardset_perlabel_list[ind].append(sample_ind)

        mean_prob_list = np.mean(prob_list)

        if mean_prob_list < conf_dict_com['hardset_thr']:
            selected_hardset_list.append(sample_ind)

        total_prob_list.append((sample_ind,prob_list))
        hardset_list[sample_ind] = mean_prob_list

    top_hardset_k = round(conf_dict_com['hardset_retaining_ratio']* len(hardset_list))

    if conf_dict_com['st_variant'] == 'hard.uniform' and conf_dict_com['hard_pick_topk']:
        selected_hardset_list = []
        pick_top = 1
        for key, value in sorted(hardset_list.items(),key=lambda x:x[1]):
            if pick_top <= top_hardset_k :
                selected_hardset_list.append(key)
                pick_top = pick_top + 1
            else:
                break
    elif conf_dict_com['st_variant'] == 'hard.label' and conf_dict_com['hard_pick_topk']:
        selected_hardset_perlabel_list = {k:[] for k in range(NUM_CLASSES)}
        for lab_ind in range(NUM_CLASSES):
            pick_top = 1
            sorted_prob_list = sorted(total_prob_list, key = lambda each_prob_list: each_prob_list[1][lab_ind])
            for p_list in sorted_prob_list:
                if pick_top <= top_hardset_k:
                    selected_hardset_perlabel_list[lab_ind].append(p_list[0])
                    pick_top = pick_top + 1
                else:
                    break

    avg_similarity = {}
    data_pos =[]
    new_data_list_diverse = []
    new_data_list_sen_diverse = []
    position_diverse = []
    new_lab_list_diverse = []
    confidence_scores = []
    original_pred_probs =[]

    with open(post_rep_filename, 'rb') as f:
        post_vec = pickle.load(f) 

    position = data_dict['testunlab_st_ind']
    for cl_ind, tup_val in enumerate(mod_op_list):
        mod_op, att_op = tup_val
        for i,pb in enumerate(mod_op):
            y_pred = np.rint(pb).astype(int)
            if sum(y_pred) == 0:
                y_pred[np.argmax(pb)] = 1
            count_labels = list(y_pred).count(1)
            if count_labels >=conf_dict_com['min_num_pred_labs'] and count_labels <=conf_dict_com['max_num_pred_labs']:
                category_coun = 0
                for j in pb:
                    if j >= conf_dict_com['confidence_thr'] or (1-j) >= conf_dict_com['confidence_thr']:
                        category_coun = category_coun + 1
                        if category_coun == NUM_CLASSES:
                            data_pos.append(position)
                            avg_dist = []
                            if conf_dict_com['st_variant'] == 'hard.label':
                                for pred_lab in range(len(y_pred)):
                                    if y_pred[pred_lab] == 1:
                                        dist_perlabel = []
                                        for pos in selected_hardset_perlabel_list[pred_lab]:
                                            dist_perlabel.append(cosine_similarity(np.array([post_vec[position]]),np.array([post_vec[pos+data_dict['train_en_ind']]])))
                                        if len(dist_perlabel) > 0:
                                            avg_dist.append(np.mean(np.array(dist_perlabel)))
                            if len(avg_dist) == 0:
                                for pos in selected_hardset_list:
                                    avg_dist.append(cosine_similarity(np.array([post_vec[position]]),np.array([post_vec[pos+data_dict['train_en_ind']]])))                               
                            avg_similarity[position] = np.mean(avg_dist)
                    else:
                        break
            position = position + 1  
    top_k = conf_dict_com['retaining_ratio'] * len(data_pos)
    
    ind = 1
    for key, value in sorted(avg_similarity.items(),key=lambda x:x[1],reverse=True):
        if ind <= top_k:
            position_diverse.append(key)
            ind = ind + 1
        else:
            break
    position_diverse.sort()

    for pos in position_diverse:
        posi = pos - data_dict['testunlab_st_ind']
        new_data_list_diverse.append(data_dict['text'][pos])
        new_data_list_sen_diverse.append(data_dict['text_sen'][pos])
        if conf_dict_com['gen_conf_scores']:
            conf_scores_persample = get_conf_scores(mod_op[posi])      
            confidence_scores.append(conf_scores_persample)
            original_pred_probs.append(mod_op[posi])
        y_pred = np.rint(mod_op[posi]).astype(int)
        if sum(y_pred) == 0:
            y_pred[np.argmax(mod_op[posi])] = 1
        new_lab_list_diverse.append(y_pred)
    pred_vals = di_op_to_label_lists(new_lab_list_diverse)

    return new_data_list_diverse, new_data_list_sen_diverse, pred_vals,position_diverse,len(new_data_list_diverse),confidence_scores,original_pred_probs

def generate_unlab_rudimentary(mod_op_list, data_dict, bac_map, prob_trans_type,iter,save_path, conf_dict_com):
    count_predicted = 0
    new_data_list = []
    new_data_list_sen =[]
    new_lab_list = []
    data_pos =[]
    confidence_scores = []
    original_pred_probs =[]

    position = data_dict['testunlab_st_ind']
    for cl_ind, tup_val in enumerate(mod_op_list):
        mod_op, att_op = tup_val
        
        for i,pb in enumerate(mod_op):
            prob_list = []
            y_pred = np.rint(pb).astype(int)
            for prob in pb:
                if prob >= 0.5:
                    prob_list.append(prob)
                else:
                    prob_list.append(1-prob)
            if sum(y_pred) == 0:
                max_val = np.argmax(pb)
                y_pred[max_val] = 1
                prob_list[max_val] = pb[max_val]
            if np.mean(prob_list) >= conf_dict_com['avg_thr']:
                count_predicted  = count_predicted + 1
                new_data_list.append(data_dict['text'][position])
                new_data_list_sen.append(data_dict['text_sen'][position])
                data_pos.append(position)
                new_lab_list.append(y_pred)
                if conf_dict_com['gen_conf_scores']:
                    conf_scores_persample = get_conf_scores(pb)      
                    confidence_scores.append(conf_scores_persample)
                    original_pred_probs.append(pb)
            position = position + 1 
    pred_vals = di_op_to_label_lists(new_lab_list)
    return new_data_list, new_data_list_sen, pred_vals,data_pos, count_predicted,confidence_scores,original_pred_probs


def evaluate_model_new(mod_op_list, data_dict, bac_map, prob_trans_type, metr_dict, metr_dict_perrun,thresh, att_flag, output_folder_name, fname_part_r_ind):
    y_pred_list = []
    true_vals = data_dict['lab'][data_dict['val_st_ind']:data_dict['val_en_ind']]
    num_test_samp = len(true_vals)
    sum_br_lists = np.zeros(num_test_samp, dtype=np.int64)
    arg_max_br_lists = np.empty(num_test_samp, dtype=np.int64)
    max_br_lists = np.zeros(num_test_samp)
    for cl_ind, tup_val in enumerate(mod_op_list):
        mod_op, att_op = tup_val
        if prob_trans_type == 'lp':
            y_pred  = np.argmax(mod_op, 1)
        elif prob_trans_type == "di":        
            y_pred = np.rint(mod_op).astype(int)
            for i in range(num_test_samp):
                if sum(y_pred[i]) == 0:
                    y_pred[i, np.argmax(mod_op[i])] = 1
        elif prob_trans_type == "dc":
            y_pred = np.zeros((mod_op.shape), dtype=np.int64)
            for ind_row, row in enumerate(mod_op):
                s_indices = np.argsort(-row)
                row_s = row[s_indices]
                dif = row_s[:len(row)-1] - row_s[1:]
                m_ind = dif.argmax()
                y_pred[ind_row, s_indices[:m_ind+1]] = 1 
        else:
            mod_op = np.squeeze(mod_op, -1)
            y_pred = np.rint(mod_op).astype(int)
            sum_br_lists += y_pred
            for i in range(num_test_samp):
                if mod_op[i] > max_br_lists[i]:
                    max_br_lists[i] = mod_op[i]
                    arg_max_br_lists[i] = cl_ind
        y_pred_list.append(y_pred)

    pred_vals = di_op_to_label_lists(y_pred_list[0])
  
    return pred_vals, true_vals, calc_metrics_print(pred_vals, true_vals, metr_dict), calc_metrics_print(pred_vals, true_vals, metr_dict_perrun)

def sent_enc_featurize_new(sent_enc_feats_raw, model_type, data_dict, poss_sent_enc_feats_emb_dict, use_saved_sent_enc_feats, save_sent_enc_feats, data_fold_path, save_fold_path, test_mode,filename):
	max_num_sent_enc_feats = 3
	max_num_attributes = 2
	sent_enc_feats = []
	var_model_hier = is_model_hier(model_type)
	sent_enc_feat_str = ''
	for sent_enc_feat_raw_dict in sent_enc_feats_raw:
		feat_name = sent_enc_feat_raw_dict['emb']
		sent_enc_feat_str += ("%s~%s~" % (sent_enc_feat_raw_dict['emb'], sent_enc_feat_raw_dict['m_id']))

		sent_enc_feat_dict ={}
		for sent_enc_feat_attr_name, sent_enc_feat_attr_val in sent_enc_feat_raw_dict.items():
			sent_enc_feat_dict[sent_enc_feat_attr_name] = sent_enc_feat_attr_val

		print("computing %s sent feats; hier model: %s; test_mode = %s" % (feat_name, var_model_hier, test_mode))

		s_filename = ("%ssent_enc_feat~%s~%s~%s.h5" % (save_fold_path, feat_name, var_model_hier,filename))
		if use_saved_sent_enc_feats and os.path.isfile(s_filename):
			print("loading %s sent feats" % feat_name)
			with h5py.File(s_filename, "r") as hf:
				sent_enc_feat_dict['feats'] = hf['feats'][:data_dict['testunlab_en_ind']]
		else:
			if var_model_hier:
				if feat_name.startswith('bert'):
					if feat_name == 'bert' or feat_name.startswith('bert_pre'):
						feats = bert_embed_posts(data_dict['text_sen'], data_dict['max_num_sent'], poss_sent_enc_feats_emb_dict[feat_name], data_fold_path)
					else:
						feats = tuned_embed_posts(data_dict['text_sen'], data_dict['max_num_sent'], poss_sent_enc_feats_emb_dict['bert'], poss_sent_enc_feats_emb_dict[feat_name], feat_name, 'bert', ("%ssent_enc_feat~bert.h5" % (save_fold_path)), bert_embed_posts, use_saved_sent_enc_feats, data_fold_path, save_fold_path)
				
			else:
				print("Error! Check the config file.")

			sent_enc_feat_dict['feats'] = feats[:len(data_dict['text'])]

			if save_sent_enc_feats:
				print("saving %s sent feats" % feat_name)
				with h5py.File(s_filename, "w") as hf:
					hf.create_dataset('feats', data=feats)

		sent_enc_feats.append(sent_enc_feat_dict)

	sent_enc_feat_str += "~" * ((max_num_sent_enc_feats - len(sent_enc_feats_raw)) * max_num_attributes)

	return sent_enc_feats, sent_enc_feat_str[:-1], s_filename

def rearrange(path_rearr,data_pos,len_of_traindata):
    regex = re.compile('train_') 
    files = os.listdir(path_rearr)
    orig_len_files = len_files = len([name for name in os.listdir(path_rearr) if os.path.isfile(os.path.join(path_rearr, name))])
    for index, file in enumerate(sorted(os.listdir(path_rearr), key=lambda x: int(x.replace(".npy", "")))):
        if index >=len_of_traindata:
            src = file
            if int(file.split('.')[0].strip()) in data_pos:
                dst = 'train_'+str(index)+".npy"
            else:
                dst = str(len_files) + ".npy"
                len_files= len_files+ 1
            os.rename(path_rearr + src,path_rearr +dst)

    len_files = orig_len_files
    len_train = len_of_traindata
    len_unlab_pos = len_of_traindata + len(data_pos)
    for index, file in enumerate(sorted(os.listdir(path_rearr), key=lambda x: int(x.replace(".npy", "").replace("train_", "")))):
        if index >=len_of_traindata:
            src = file
            if(regex.search(file.split('.')[0].strip()) != None):
                dst = str(len_train) + ".npy"
                len_train = len_train + 1
            elif int(file.split('.')[0].strip()) >= (len_files):
                dst = str(len_unlab_pos) + ".npy"
                len_unlab_pos = len_unlab_pos + 1
            os.rename(path_rearr + src,path_rearr +dst) 

def rearr_bert(data_pos,len_of_traindata,s_filename):
    indexes = list(range(len(data_dict['text'])))
    indexes = [x for x in indexes if x not in data_pos]
    indexes[len_of_traindata:len_of_traindata] = data_pos
    with h5py.File(s_filename, "r") as hf:
        bert_feat = hf['feats'][:data_dict['testunlab_en_ind']]
    bert_edited = [bert_feat[i] for i in indexes]
    with h5py.File(s_filename, "w") as hfile:
        hfile.create_dataset('feats', data=bert_edited)  

def rearr_postvec(data_pos,len_of_traindata,s_filename):
    with open(s_filename, 'rb') as f:
    	postveclist = pickle.load(f) 
    indexes = list(range(len(data_dict['text'])))
    indexes = [x for x in indexes if x not in data_pos]
    indexes[len_of_traindata:len_of_traindata] = data_pos
    postveclist_edited = [postveclist[i] for i in indexes]
    os.remove(s_filename)
    with open(s_filename, 'wb') as f1:
    	pickle.dump(postveclist_edited, f1) 


def prep_unlab(iter, data_dict, conf_dict_com, conf_dict_list, post_vecs,conf_scores_array,original_probs_array,initial_train_en_ind):
    for conf_dict in conf_dict_list:
        for prob_trans_type in conf_dict["prob_trans_types"]:
            trainY_list, trainY_noncat_list, num_classes_var, bac_map = transform_labels(data_dict['lab'][:data_dict['train_en_ind']], prob_trans_type, conf_dict_com['test_mode'], conf_dict_com["save_folder_name"],conf_dict_com['total_data_filename'],None)
            for class_imb_flag in conf_dict["class_imb_flags"]:
                loss_func_list, nonlin, out_vec_size, cw_list = class_imb_loss_nonlin(trainY_noncat_list, class_imb_flag, num_classes_var, prob_trans_type, conf_dict_com['test_mode'], conf_dict_com["save_folder_name"],conf_dict_com['use_conf_scores'],None,None, None, None,None,[],[])
                for model_type in conf_dict["model_types"]:
                    for word_feats_raw in conf_dict["word_feats_l"]:
                        word_feats, word_feat_str = word_featurize(word_feats_raw, model_type, data_dict, conf_dict_com['poss_word_feats_emb_dict'], conf_dict_com['use_saved_word_feats'], conf_dict_com['save_word_feats'], conf_dict_com["data_folder_name"], conf_dict_com["save_folder_name"], conf_dict_com["test_mode"],conf_dict_com['total_data_filename'])
                        for sent_enc_feats_raw in conf_dict["sent_enc_feats_l"]:
                            sent_enc_feats, sent_enc_feat_str, s_filename = sent_enc_featurize_new(sent_enc_feats_raw, model_type, data_dict, conf_dict_com['poss_sent_enc_feats_emb_dict'], conf_dict_com['use_saved_sent_enc_feats'], conf_dict_com['save_sent_enc_feats'], conf_dict_com["data_folder_name"], conf_dict_com["save_folder_name"], conf_dict_com["test_mode"],conf_dict_com['total_data_filename'])
                            for num_cnn_filters in conf_dict["num_cnn_filters"]:
                                for max_pool_k_val in conf_dict["max_pool_k_vals"]:
                                    for cnn_kernel_set in conf_dict["cnn_kernel_sets"]:
                                        cnn_kernel_set_str = str(cnn_kernel_set)[1:-1].replace(',','').replace(' ', '')
                                        for rnn_type in conf_dict["rnn_types"]:
                                            for rnn_dim in conf_dict["rnn_dims"]:
                                                for att_dim in conf_dict["att_dims"]:
                                                    for stack_rnn_flag in conf_dict["stack_rnn_flags"]:
                                                        mod_op_list_save_list = []
                                                        for thresh in conf_dict["threshes"]:                                                            
                                                            post_rep_filename = ("%s%s~%s~%d.pickle" % (conf_dict_com['save_folder_name'], 'raw_postvecs',model_type, 0))
                                                            if (conf_dict_com['st_variant'].startswith('combine') or conf_dict_com['st_variant'].startswith('diversity') or conf_dict_com['st_variant'].startswith('intersection')) and iter == 0 and not os.path.isfile(post_rep_filename):
                                                            	test_st =0
                                                            	test_en = data_dict['testunlab_en_ind']
                                                            	post_vec_op = gen_postvecs('mod~0~1.h5',test_st,test_en,word_feats, sent_enc_feats, data_dict, model_type, num_cnn_filters, rnn_type, fname_part, loss_func_list[0], cw_list[0], nonlin, out_vec_size, rnn_dim, att_dim, max_pool_k_val,stack_rnn_flag,cnn_kernel_set, conf_dict_com["save_folder_name"], conf_dict_com["use_saved_model"], conf_dict_com["gen_att"], conf_dict_com["LEARN_RATE"], conf_dict_com["dropO1"], conf_dict_com["dropO2"], conf_dict_com["BATCH_SIZE"], conf_dict_com["EPOCHS"], conf_dict_com["save_model"],conf_dict_com['st_variant'])
                                                            	with open(post_rep_filename, 'wb') as f:
                                                                	pickle.dump(post_vec_op, f)                                                            
                                                            len_data_train = data_dict['train_en_ind']
                                                            metr_dict = init_metr_dict()
                                                            info_str = "iter: %s, model: %s, word_feats = %s, sent_enc_feats = %s, prob_trans_type = %s, class_imb_flag = %s, num_cnn_filters = %s, cnn_kernel_set = %s, rnn_type = %s, rnn_dim = %s, att_dim = %s, max_pool_k_val = %s, stack_rnn_flag = %s, thresh = %s, test mode = %s, st varinat = %s, conf thr =%f, retaining ratio = %f, hardset thr = %f, hardset perlabelthr = %f, hardset retaining ratio = %f, avg thr = %f " % (iter,model_type,word_feat_str,sent_enc_feat_str,prob_trans_type,class_imb_flag,num_cnn_filters,cnn_kernel_set,rnn_type,rnn_dim,att_dim,max_pool_k_val,stack_rnn_flag, thresh, conf_dict_com["test_mode"], conf_dict_com['st_variant'],conf_dict_com['confidence_thr'],conf_dict_com['retaining_ratio'], conf_dict_com['hardset_thr'], conf_dict_com['hardset_perlabel_thr'], conf_dict_com['hardset_retaining_ratio'],conf_dict_com['avg_thr'])
                                                            fname_part = ("%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%f~%f~%f~%f~%f~%f~%s~%s" % (model_type,word_feat_str,sent_enc_feat_str,prob_trans_type,class_imb_flag,num_cnn_filters,cnn_kernel_set_str,rnn_type,rnn_dim,att_dim,max_pool_k_val,stack_rnn_flag, conf_dict_com['st_variant'],conf_dict_com['confidence_thr'],conf_dict_com['retaining_ratio'], conf_dict_com['hardset_thr'], conf_dict_com['hardset_perlabel_thr'], conf_dict_com['hardset_retaining_ratio'], conf_dict_com['avg_thr'], conf_dict_com["test_mode"],iter))
                                                            avgf1inst_f1mac_init = 0.0
                                                            for run_ind in range(conf_dict_com["num_runs"]):
                                                                metr_dict_perrun = init_metr_dict()
                                                                mod_op_list = []
                                                                train_en = data_dict['train_en_ind']
                                                                test_st =data_dict['val_st_ind']
                                                                test_en = data_dict['val_en_ind']
                                                                for m_ind, (loss_func, cw, trainY) in enumerate(zip(loss_func_list, cw_list, trainY_list)):
                                                                        mod_op, att_op = train_predict_new('val',train_en,test_st,test_en,word_feats, sent_enc_feats, trainY, data_dict, model_type, num_cnn_filters, rnn_type, fname_part, loss_func, cw, nonlin, out_vec_size, rnn_dim, att_dim, max_pool_k_val,stack_rnn_flag,cnn_kernel_set, m_ind, run_ind, conf_dict_com["save_folder_name"], conf_dict_com["use_saved_model"], conf_dict_com["gen_att"], conf_dict_com["LEARN_RATE"], conf_dict_com["dropO1"], conf_dict_com["dropO2"], conf_dict_com["BATCH_SIZE"], conf_dict_com["EPOCHS"], conf_dict_com["save_model"], conf_dict_com['bert_max_seq_len'], conf_dict_com["st_variant"],conf_scores_array,conf_dict_com['use_conf_scores'])
                                                                        mod_op_list.append((mod_op, att_op))
                                                                pred_vals, true_vals, metr_dict,metr_dict_perrun = evaluate_model_new(mod_op_list, data_dict, bac_map, prob_trans_type, metr_dict, metr_dict_perrun, thresh, conf_dict_com['gen_att'], conf_dict_com["output_folder_name"], ("%s_%d" % (fname_part,run_ind)))
                                                                if conf_dict_com['gen_inst_res'] and run_ind == 0:
                                                                    insights_results(pred_vals, true_vals, data_dict['text'][data_dict['val_st_ind']:data_dict['val_en_ind']], data_dict['text_sen'][data_dict['val_st_ind']:data_dict['val_en_ind']], data_dict['lab'][0:data_dict['train_en_ind']], fname_part, conf_dict_com["output_folder_name"])   
                                                                metr_dict_perrun = aggregate_metr(metr_dict_perrun, 1)
                                                                avgf1inst_f1mac = (metr_dict_perrun['avg_fl_ma']+metr_dict_perrun['avg_fi'])/2
                                                                if avgf1inst_f1mac > avgf1inst_f1mac_init:
                                                                    avgf1inst_f1mac_init = avgf1inst_f1mac
                                                                    best_run = run_ind
                                                            f_res.write("%s\n\n" % info_str)
                                                            print("%s\n" % info_str)
                                                            metr_dict = aggregate_metr(metr_dict, conf_dict_com["num_runs"])                                                           
                                                            write_results(metr_dict, f_res)  
                                                            if iter != conf_dict_com["num_iter"] - 1 :                                                                                                                                    
                                                                mod_op_list = []
                                                                if conf_dict_com['st_variant'].startswith('hard'):                                                               
                                                                    train_en = held_train_st = data_dict['train_en_ind']
                                                                    held_train_en = data_dict['held_train_en_ind']
                                                                    heldout_lab = data_dict['lab'][held_train_st:held_train_en]
                                                                    heldout_op, heldout_att_op = train_predict_new('heldout',train_en,held_train_st,held_train_en,word_feats, sent_enc_feats, trainY, data_dict, model_type, num_cnn_filters, rnn_type, fname_part, loss_func, cw, nonlin, out_vec_size, rnn_dim, att_dim, max_pool_k_val,stack_rnn_flag,cnn_kernel_set,m_ind, best_run, conf_dict_com["save_folder_name"], conf_dict_com["use_saved_model"], conf_dict_com["gen_att"], conf_dict_com["LEARN_RATE"], conf_dict_com["dropO1"], conf_dict_com["dropO2"], conf_dict_com["BATCH_SIZE"], conf_dict_com["EPOCHS"], conf_dict_com["save_model"], conf_dict_com['bert_max_seq_len'], conf_dict_com["st_variant"],conf_scores_array,conf_dict_com['use_conf_scores'])
                                                                train_en = data_dict['train_en_ind']
                                                                test_st = data_dict['testunlab_st_ind']
                                                                test_en = data_dict['testunlab_en_ind']
                                                                mod_op, att_op = train_predict_new('gen_unlab',train_en,test_st,test_en,word_feats, sent_enc_feats, trainY, data_dict, model_type, num_cnn_filters, rnn_type, fname_part, loss_func, cw, nonlin, out_vec_size, rnn_dim, att_dim, max_pool_k_val,stack_rnn_flag,cnn_kernel_set, m_ind, best_run, conf_dict_com["save_folder_name"], conf_dict_com["use_saved_model"], conf_dict_com["gen_att"], conf_dict_com["LEARN_RATE"], conf_dict_com["dropO1"], conf_dict_com["dropO2"], conf_dict_com["BATCH_SIZE"], conf_dict_com["EPOCHS"], conf_dict_com["save_model"], conf_dict_com['bert_max_seq_len'], conf_dict_com["st_variant"],conf_scores_array,conf_dict_com['use_conf_scores'])
                                                                mod_op_list.append((mod_op, att_op))
                                                                if conf_dict_com['st_variant'].startswith('basic'):
                                                                    new_data, new_data_sen, new_lab,data_pos, predicted_ex,conf_scores,original_pred_probs = generate_unlab_IPSPC(mod_op_list, data_dict, bac_map, prob_trans_type,iter,conf_dict_com['save_folder_name'], conf_dict_com)
                                                                elif conf_dict_com['st_variant'] == 'conf_thr':
                                                                    new_data, new_data_sen, new_lab,data_pos, predicted_ex,conf_scores,original_pred_probs = generate_unlab_basic(mod_op_list, data_dict, bac_map, prob_trans_type,iter,conf_dict_com['save_folder_name'], conf_dict_com)
                                                                elif conf_dict_com['st_variant'].startswith('diversity'):
                                                                    new_data, new_data_sen, new_lab,data_pos, predicted_ex,conf_scores,original_pred_probs = generate_unlab_diversity(mod_op_list, data_dict, bac_map, prob_trans_type,iter,conf_dict_com['save_folder_name'], conf_dict_com,post_rep_filename)
                                                                elif conf_dict_com['st_variant'].startswith('support'):
                                                                    new_data, new_data_sen, new_lab,data_pos, predicted_ex,conf_scores,original_pred_probs = generate_unlab_favour(mod_op_list, data_dict, bac_map, prob_trans_type,iter,conf_dict_com['save_folder_name'], conf_dict_com)
                                                                elif conf_dict_com['st_variant'].startswith('combine'):
                                                                    new_data, new_data_sen, new_lab,data_pos, predicted_ex,conf_scores,original_pred_probs = generate_unlab_diversity_favour(mod_op_list, data_dict, bac_map, prob_trans_type,iter,conf_dict_com['save_folder_name'],conf_dict_com,post_rep_filename)
                                                                elif conf_dict_com['st_variant'].startswith('intersection'):
                                                                    new_data, new_data_sen, new_lab,data_pos, predicted_ex,conf_scores,original_pred_probs =  generate_unlab_diversity_favour_intersection(mod_op_list, data_dict, bac_map, prob_trans_type,iter,conf_dict_com['save_folder_name'],conf_dict_com,post_rep_filename)                                                              
                                                                elif conf_dict_com['st_variant'].startswith('random'):
                                                                    new_data, new_data_sen, new_lab,data_pos, predicted_ex,conf_scores,original_pred_probs = generate_unlab_random(mod_op_list, data_dict, bac_map, prob_trans_type,iter,conf_dict_com['save_folder_name'],conf_dict_com)
                                                                elif conf_dict_com['st_variant'].startswith('hard'):
                                                                    new_data, new_data_sen, new_lab,data_pos, predicted_ex,conf_scores,original_pred_probs = generate_unlab_hard(heldout_op,heldout_lab,mod_op_list, data_dict, bac_map, prob_trans_type,iter,conf_dict_com['save_folder_name'],conf_dict_com,post_rep_filename)
                                                                elif conf_dict_com['st_variant'] == 'rudimentary':
                                                                    new_data, new_data_sen, new_lab,data_pos, predicted_ex,conf_scores,original_pred_probs = generate_unlab_rudimentary(mod_op_list, data_dict, bac_map, prob_trans_type,iter,conf_dict_com['save_folder_name'], conf_dict_com)
                                                                f_tsv.write("%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s\t%s\t%s\t%s\t%s\t%s\n" % (best_run,conf_dict_com['st_variant'], conf_dict_com['max_num_pred_labs'], conf_dict_com['confidence_thr'], conf_dict_com['retaining_ratio'],conf_dict_com['hardset_thr'], conf_dict_com['hardset_perlabel_thr'],conf_dict_com['hardset_retaining_ratio'],conf_dict_com['avg_thr'], iter,predicted_ex ,model_type,word_feat_str,sent_enc_feat_str,prob_trans_type,class_imb_flag,num_cnn_filters,cnn_kernel_set_str,thresh,rnn_dim,att_dim,max_pool_k_val,stack_rnn_flag,(metr_dict['avg_fl_ma']+metr_dict['avg_fi'])/2,(metr_dict['std_fl_ma']+metr_dict['std_fi'])/2,metr_dict['avg_fi'],metr_dict['avg_fl_ma'],(metr_dict['avg_fl_ma']+metr_dict['avg_fi']+metr_dict['avg_ji']+metr_dict['avg_fl_mi'])/4,metr_dict['avg_ji'],metr_dict['avg_fl_mi'],metr_dict['avg_em'],metr_dict['avg_ihl'],rnn_type,conf_dict_com["LEARN_RATE"],conf_dict_com["BATCH_SIZE"],conf_dict_com["dropO1"],conf_dict_com["dropO2"], conf_dict_com["test_mode"]))
                                                                data_dict['text'] = [data_dict['text'][i] for i in range(len(data_dict['text'])) if i not in data_pos]
                                                                data_dict['text'][data_dict['train_en_ind']:data_dict['train_en_ind']] = new_data
                                                                data_dict['text_sen'] = [data_dict['text_sen'][i] for i in range(len(data_dict['text_sen'])) if i not in data_pos]
                                                                data_dict['text_sen'][data_dict['train_en_ind']:data_dict['train_en_ind']] = new_data_sen
                                                                data_dict['lab'][data_dict['train_en_ind']:data_dict['train_en_ind']] = new_lab
                                                                if conf_dict_com['gen_conf_scores']: 
                                                                    conf_scores_array = np.insert(conf_scores_array,data_dict['train_en_ind'],conf_scores,axis=0)
                                                                    original_probs_array = np.insert(original_probs_array,len(original_probs_array),original_pred_probs,axis =0)
                                                                if conf_dict_com['st_variant'].startswith('hard'):
                                                                    data_dict['train_en_ind'] = data_dict['held_train_st_ind'] = data_dict['train_en_ind'] + len(new_data)
                                                                    data_dict['held_train_en_ind'] = data_dict['val_st_ind'] = data_dict['held_train_en_ind'] + len(new_data)
                                                                else:
                                                                    data_dict['train_en_ind'] = data_dict['val_st_ind'] = data_dict['train_en_ind'] + len(new_data)
                                                                data_dict['val_en_ind'] = data_dict['test_st_ind'] = data_dict['val_en_ind'] + len(new_data)
                                                                data_dict['test_en_ind'] = data_dict['testunlab_st_ind'] = data_dict['test_en_ind'] + len(new_data)                                                                   
                                                                data_dict['testunlab_en_ind'] = len(data_dict['text'])   
                                                                data_dict_filename = ("%s%s_%s_%f_%f_%f_%f_%f_%f_%d.pickle" % (conf_dict_com["save_folder_name"], 'raw_data', conf_dict_com['st_variant'], conf_dict_com['confidence_thr'], conf_dict_com['retaining_ratio'],conf_dict_com['hardset_thr'],conf_dict_com['hardset_perlabel_thr'], conf_dict_com['hardset_retaining_ratio'], conf_dict_com['avg_thr'], iter+1))
                                                                index_filename = ("%s%s_%s_%f_%f_%f_%f_%f_%f_%d.txt" % (conf_dict_com["save_folder_name"], 'index', conf_dict_com['st_variant'],conf_dict_com['confidence_thr'], conf_dict_com['retaining_ratio'],conf_dict_com['hardset_thr'],conf_dict_com['hardset_perlabel_thr'], conf_dict_com['hardset_retaining_ratio'],conf_dict_com['avg_thr'],iter+1))

                                                                with open(index_filename, 'w') as f_in:
                                                                	for i in range(len(data_dict['text'])):
                                                                		ind = data_dict['tot_data'].index(data_dict['text'][i])
                                                                		f_in.write(str(ind))
                                                                		f_in.write('\n')
                                                                with open(data_dict_filename, 'wb') as f_data:
                                                                	pickle.dump(data_dict, f_data)

                                                                if conf_dict_com['gen_conf_scores']:
	                                                                conf_score_filename = ("%s%s_%s_%f_%f_%f_%f_%f_%f_%d.pickle" % (conf_dict_com["save_folder_name"], 'raw_conf_scores',conf_dict_com['st_variant'],conf_dict_com['confidence_thr'], conf_dict_com['retaining_ratio'],conf_dict_com['hardset_thr'], conf_dict_com['hardset_perlabel_thr'],conf_dict_com['hardset_retaining_ratio'],conf_dict_com['avg_thr'], iter+1))
	                                                                with open(conf_score_filename, 'wb') as conf_data:
	                                                                    pickle.dump([conf_scores_array,initial_train_en_ind,data_dict['train_en_ind']], conf_data,)
	                                                                prob_filename = ("%s%s_%s_%f_%f_%f_%f_%f_%f_%d.pickle" % (conf_dict_com["save_folder_name"], 'raw_original_probs',conf_dict_com['st_variant'],conf_dict_com['confidence_thr'], conf_dict_com['retaining_ratio'],conf_dict_com['hardset_thr'], conf_dict_com['hardset_perlabel_thr'],conf_dict_com['hardset_retaining_ratio'],conf_dict_com['avg_thr'],iter+1))
	                                                                with open(prob_filename, 'wb') as prob_data:
	                                                                    pickle.dump(original_probs_array, prob_data)
                                                                return data_pos,s_filename, len_data_train,post_rep_filename,conf_scores_array, original_probs_array
                                                            else:
                                                                predicted_ex = 0
                                                                f_tsv.write("%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s\t%s\t%s\t%s\t%s\t%s\n" % (best_run,conf_dict_com['st_variant'], conf_dict_com['max_num_pred_labs'], conf_dict_com['confidence_thr'], conf_dict_com['retaining_ratio'],conf_dict_com['hardset_thr'],conf_dict_com['hardset_perlabel_thr'], conf_dict_com['hardset_retaining_ratio'],conf_dict_com['avg_thr']. iter,predicted_ex ,model_type,word_feat_str,sent_enc_feat_str,prob_trans_type,class_imb_flag,num_cnn_filters,cnn_kernel_set_str,thresh,rnn_dim,att_dim,max_pool_k_val,stack_rnn_flag,(metr_dict['avg_fl_ma']+metr_dict['avg_fi'])/2,(metr_dict['std_fl_ma']+metr_dict['std_fi'])/2,metr_dict['avg_fi'],metr_dict['avg_fl_ma'],(metr_dict['avg_fl_ma']+metr_dict['avg_fi']+metr_dict['avg_ji']+metr_dict['avg_fl_mi'])/4,metr_dict['avg_ji'],metr_dict['avg_fl_mi'],metr_dict['avg_em'],metr_dict['avg_ihl'],rnn_type,conf_dict_com["LEARN_RATE"],conf_dict_com["BATCH_SIZE"],conf_dict_com["dropO1"],conf_dict_com["dropO2"], conf_dict_com["test_mode"]))
                                                                data_pos = []
                                                                return data_pos,s_filename, len_data_train, post_rep_filename, conf_scores_array, original_probs_array

f_res = open(conf_dict_com["output_folder_name"] + '/' + conf_dict_com["res_filename"], 'a')
tsv_path = conf_dict_com["output_folder_name"] + '/' + conf_dict_com["res_tsv_filename"]
if os.path.isfile(tsv_path):
    f_tsv = open(tsv_path, 'a')
else:
    f_tsv = open(tsv_path, 'w')
    f_tsv.write("Best_run\tself_training_variant\t minmax_pred_labels\tconfidence_thr\tretaining_ratio\titer\t pred_noof_samples\tmodel\tword feats\tsent feats\ttrans\tclass imb\tcnn fils\tcnn kernerls\tthresh\trnn dim\tatt dim\tpool k\tstack RNN\tf_I+f_Ma\tstd_d\tf1-Inst\tf1-Macro\tsum_4\tJaccard\tf1-Micro\tExact\tI-Ham\trnn type\tl rate\tb size\tdr1\tdr2\ttest mode\n") 

data_dict, data_dict_filename,cl_in_filename = load_data_new(conf_dict_com["filename"], conf_dict_com["data_folder_name"], conf_dict_com["save_folder_name"], conf_dict_com['TEST_RATIO'], conf_dict_com['VALID_RATIO'], conf_dict_com['RANDOM_STATE'], conf_dict_com['MAX_WORDS_SENT'], conf_dict_com["test_mode"])
initial_train_en_ind  = data_dict['train_en_ind']
print("max # sentences: %d, max # words per sentence: %d, max # words per post: %d" % (data_dict['max_num_sent'], data_dict['max_words_sent'], data_dict['max_post_length']))
conf_scores_array = np.ones([data_dict['val_en_ind'], NUM_CLASSES])
original_probs_array = np.zeros([0,NUM_CLASSES])
startTime = time.time()

post_vecs = None
for i in range(conf_dict_com["num_iter"]):
    print ("***********Iteration number**************", i)
    data_pos,s_filename, len_data_train,post_rep_filename,conf_scores_array,original_probs_array = prep_unlab(i, data_dict, conf_dict_com, conf_dict_list, post_vecs,conf_scores_array,original_probs_array,initial_train_en_ind)
    path_rearr_elmo = conf_dict_com['save_folder_name'] + 'word_vecs~elmo~totalDat/False/'   
    if i != conf_dict_com["num_iter"] - 1 :
        if conf_dict_com['st_variant'].startswith('combine') or conf_dict_com['st_variant'].startswith('diversity') or conf_dict_com['st_variant'].startswith('intersection'):
            rearr_postvec(data_pos,len_data_train,post_rep_filename)
        rearr_bert(data_pos,len_data_train,s_filename)
        rearrange(path_rearr_elmo,data_pos,len_data_train)
        list_saved_folds = os.listdir(conf_dict_com['save_folder_name'])
        for item in list_saved_folds:
            if item.endswith(".pickle") and item[0] != 'r' and item != "comb_vocab~glove.pickle":
                os.remove(os.path.join(conf_dict_com['save_folder_name'], item))
    timeLapsed = int(time.time() - startTime + 0.5)
    hrs = timeLapsed/3600.
    t_str = "%.1f hours = %.1f minutes over %d hours\n" % (hrs, (timeLapsed % 3600)/60.0, int(hrs))
    print(t_str)
    f_res.write("%s\n" % t_str)
