import os
import numpy as np
from dlModels import get_model, attLayer_hier, sep_corr_multi_binary_loss, fuzzy_l1_corr_multi_binary_loss,fuzzy_corr_multi_binary_loss, unnorm_corr_multi_binary_loss,corr_multi_binary_loss,multi_binary_loss, multi_binary_loss_conf,kl_divergence_loss,mse_loss,br_binary_loss, lp_categ_loss, multi_cat_w_loss, multi_cat_loss, kmax_pooling
from sklearn.utils import class_weight
from loadPreProc import *
from evalMeasures import *
from keras.utils import to_categorical
from keras import backend as K
from gen_batch_keras import TrainGenerator, TestGenerator
import pickle
import json
import tensorflow as tf

def evaluate_model(mod_op_list, data_dict, bac_map, prob_trans_type, metr_dict, thresh, att_flag, output_folder_name, fname_part_r_ind):
    y_pred_list = []
    true_vals = data_dict['lab'][data_dict['test_st_ind']:data_dict['test_en_ind']]
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

    if prob_trans_type == 'lp':
        pred_vals = powerset_vec_to_label_lists(y_pred_list[0], bac_map)
    elif prob_trans_type == "di" or prob_trans_type == "dc":        
        pred_vals = di_op_to_label_lists(y_pred_list[0])
    else:
        for i in range(len(true_vals)):
            if sum_br_lists[i] == 0:
                y_pred_list[arg_max_br_lists[i]][i] = 1
        pred_vals = br_op_to_label_lists(y_pred_list)
  
    if att_flag:
        true_vals_multi_hot = trans_labels_multi_hot(true_vals)
        for ind, data_ind in enumerate(range(data_dict['test_st_ind'], data_dict['test_en_ind'])):
            att_path = "%satt_info/%s/" % (output_folder_name, fname_part_r_ind)
            os.makedirs(att_path, exist_ok=True)
            true_vals_multi_hot_ind = true_vals_multi_hot[ind].tolist()
            y_pred_list_0_ind = y_pred_list[0][ind].tolist()
            mod_op_list_0_0_ind = mod_op_list[0][0][ind].tolist()
            post_sens = data_dict['text_sen'][data_ind][:data_dict['max_num_sent']]
            post_sens_split = []
            for sen in post_sens:
                post_sens_split.append(sen.split(' '))
            for clust_ind, att_arr in enumerate(mod_op_list[0][1]):
                att_list = []
                if len(att_arr.shape) == 3:
                    fname_att = ("%s%d~w%d.json" % (att_path, ind, clust_ind))
                    for ind_sen, split_sen in enumerate(post_sens_split):
                        my_sen_dict = {}
                        my_sen_dict['text'] = split_sen
                        my_sen_dict['label'] = true_vals_multi_hot_ind
                        my_sen_dict['prediction'] = y_pred_list_0_ind
                        my_sen_dict['posterior'] = mod_op_list_0_0_ind
                        my_sen_dict['attention'] = att_arr[ind, ind_sen, :].tolist()
                        my_sen_dict['id'] = "%d~w%d~%d" % (ind, clust_ind, ind_sen)
                        att_list.append(my_sen_dict)
                else:
                    my_sen_dict = {}
                    my_sen_dict['label'] = true_vals_multi_hot_ind
                    my_sen_dict['prediction'] = y_pred_list_0_ind
                    my_sen_dict['posterior'] = mod_op_list_0_0_ind
                    my_sen_dict['attention'] = att_arr[ind, :].tolist()
                    if fname_part_r_ind.startswith('hier_fuse'):
                        fname_att = ("%s%d~s%d.json" % (att_path, ind, clust_ind))
                        my_sen_dict['text'] = data_dict['text_sen'][data_ind]
                        my_sen_dict['id'] = "%d~s%d" % (ind, clust_ind)
                    else:
                        fname_att = ("%s%d~w%d.json" % (att_path, ind, clust_ind))
                        my_sen_dict['text'] = data_dict['text'][data_ind]
                        my_sen_dict['id'] = "%d~w%d" % (ind, clust_ind)
                    att_list.append(my_sen_dict)
                with open(fname_att, 'w') as f:
                    json.dump(att_list, f)

    return pred_vals, true_vals, calc_metrics_print(pred_vals, true_vals, metr_dict)


def train_predict(word_feats, sent_enc_feats, trainY, data_dict, model_type, num_cnn_filters, rnn_type, fname_part, loss_func, class_w, nonlin, out_vec_size, rnn_dim, att_dim, max_pool_k_val, stack_rnn_flag, cnn_kernel_set, m_ind, run_ind, save_folder_name, use_saved_model, gen_att, learn_rate, dropO1, dropO2, batch_size, num_epochs, save_model,st_variant,conf_scores_array,use_conf_scores,multi_task_tl,share_weights_sep_mt,single_inp_tasks_list,sep_inp_tasks_list):
    att_op = None
    fname_mod_op = ("%s%s/iop~%d~%d.pickle" % (save_folder_name, fname_part, m_ind, run_ind))
    if use_saved_model and os.path.isfile(fname_mod_op):
        print("loading model o/p")
        with open(fname_mod_op, 'rb') as f:
            mod_op = pickle.load(f)
        if gen_att:
            fname_att_op = ("%s%s/att_op~%d~%d.pickle" % (save_folder_name, fname_part, m_ind, run_ind))
            if os.path.isfile(fname_att_op):
                with open(fname_att_op, 'rb') as f:
                    att_op = pickle.load(f)
    else:
        model, model_multi_task_or_tr, att_mod, post_vec = get_model(model_type, data_dict['max_post_length'], data_dict['max_num_sent'], data_dict['max_words_sent'], word_feats, sent_enc_feats, learn_rate, dropO1, dropO2, num_cnn_filters, rnn_type, loss_func, nonlin, out_vec_size, rnn_dim, att_dim, max_pool_k_val, stack_rnn_flag, cnn_kernel_set,st_variant,use_conf_scores,multi_task_tl,share_weights_sep_mt,single_inp_tasks_list,sep_inp_tasks_list)
        fname_mod = ("%s%s/mod~%d~%d.h5" % (save_folder_name, fname_part, m_ind, run_ind))
        if use_saved_model and os.path.isfile(fname_mod):
            print("loading model")
            model.load_weights(fname_mod)
        else:      
            if multi_task_tl ==  "tr_learn":
                training_generator_aux = TrainGenerator(np.arange(0, data_dict['train_en_ind']),trainY,word_feats,sent_enc_feats, data_dict, batch_size,conf_scores_array,use_conf_scores,multi_task_tl,True,single_inp_tasks_list,sep_inp_tasks_list)
                model_multi_task_or_tr.fit_generator(generator=training_generator_aux, epochs=num_epochs, shuffle=False, verbose=1, use_multiprocessing=False, workers=1)
            training_generator = TrainGenerator(np.arange(0, data_dict['train_en_ind']),trainY,word_feats, sent_enc_feats, data_dict, batch_size,conf_scores_array,use_conf_scores,multi_task_tl,False,single_inp_tasks_list,sep_inp_tasks_list)
            if multi_task_tl == "multi_task":
                model_multi_task_or_tr.fit_generator(generator=training_generator, epochs=num_epochs, shuffle=False, verbose=1, use_multiprocessing=False, workers=1)
            else:
                model.fit_generator(generator=training_generator, epochs=num_epochs, shuffle=False, verbose=1, use_multiprocessing=False, workers=1)
            if save_model:    
                print("saving model")
                os.makedirs(save_folder_name + fname_part, exist_ok=True)
                model.save(fname_mod)
                if multi_task_tl == "multi_task" or multi_task_tl == "tr_learn":
                    fname_mod_tr_learn = ("%s%s/mod_mt_tr_l~%d~%d.h5" % (save_folder_name, fname_part, m_ind, run_ind))
                    print("saving multi_task/tr_learn model")
                    model_multi_task_or_tr.save(fname_mod_tr_learn)
        test_generator = TestGenerator(np.arange(data_dict['test_st_ind'], data_dict['test_en_ind']), word_feats, sent_enc_feats, data_dict, batch_size, conf_scores_array, use_conf_scores)
        mod_op = model.predict_generator(generator=test_generator, verbose=1, use_multiprocessing=True, workers=2)
        if gen_att and (att_mod is not None):
            os.makedirs(save_folder_name + fname_part, exist_ok=True)
            att_op = att_mod.predict_generator(generator=test_generator, verbose=1, use_multiprocessing=True, workers=2)
            if type(att_op) != list:
                att_op = [att_op]
            fname_att_op = ("%s%s/att_op~%d~%d.pickle" % (save_folder_name, fname_part, m_ind, run_ind))
            with open(fname_att_op, 'wb') as f:
                pickle.dump(att_op, f)
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

def get_lossfunc_weights(loss_func,classi_loss_weight,single_inp_tasks_list,sep_inp_tasks_list):
    losses = {"classi": loss_func}
    lossweights = {"classi":classi_loss_weight}   
    for single_task_tup in single_inp_tasks_list:
        single_task_name, single_task_dict = single_task_tup
        if single_task_name == "topic":
            losses.update({"topic": single_task_dict['loss_func']})
        elif single_task_name == "kmeans":
            losses.update({"kmeans":lp_categ_loss(single_task_dict['cw_aux'])})
        lossweights.update({single_task_name:single_task_dict['loss_weight']})
    for sep_task_tup in sep_inp_tasks_list:
        sep_task_name, sep_task_dict = sep_task_tup
        if sep_task_name == "sd":
            losses.update({"sd":'binary_crossentropy'})
        elif sep_task_name == "sepkmeans":
            losses.update({"sepkmeans":lp_categ_loss(sep_task_dict['cw_aux'])})
        elif sep_task_name == "septopics":
            losses.update({"septopics":sep_task_dict['loss_func']})
        lossweights.update({sep_task_name:sep_task_dict['loss_weight']})
    return losses, lossweights

def class_imb_loss_nonlin(trainY_noncat_list, class_imb_flag, num_classes_var, prob_trans_type, test_mode, save_fold_path,use_conf_scores,multi_task_tl,classi_loss_weight,prime_filename,aux_filename_str, augment_data,single_inp_tasks_list,sep_inp_tasks_list,uncorr_c_pairs_filename,beta,label_corr_setting,train_ratio):
    filename = "%sclass_imb~%s~%s~%s~%s~%s~%s~%s~%s.pickle" % (save_fold_path, class_imb_flag, prob_trans_type, test_mode,prime_filename,aux_filename_str,augment_data,multi_task_tl,train_ratio)
    loss_func_list = []
    if os.path.isfile(filename):
        print("loading class imb for %s and %s; test mode = %s; prime_filename = %s; aux_filename = %s; augment_data = %s" % (class_imb_flag, prob_trans_type, test_mode,prime_filename,aux_filename_str,augment_data))
        with open(filename, 'rb') as f:
            nonlin, out_vec_size, cw_list, single_inp_list,sep_inp_list = pickle.load(f)
            for ind,single_task_tup in enumerate(single_inp_tasks_list):
                single_task_name, single_task_dict = single_task_tup
                single_task_dict['out_vec_size'] = single_inp_list[ind][0]
                single_task_dict['nonlin'] = single_inp_list[ind][1]
                single_task_dict['cw_aux'] = single_inp_list[ind][2]
            for ind, sep_task_tup in enumerate(sep_inp_tasks_list):
                sep_task_name, sep_task_dict = sep_task_tup
                sep_task_dict['out_vec_size'] = sep_inp_list[ind][0]
                sep_task_dict['nonlin'] = sep_inp_list[ind][1]
                sep_task_dict['cw_aux'] = sep_inp_list[ind][2]
    else:
        sep_inp_list =[]
        single_inp_list = []
        print("computing class imb for %s and %s; test mode = %s; prime_filename = %s; aux_filename = %s; augment_data = %s" % (class_imb_flag, prob_trans_type, test_mode,prime_filename,aux_filename_str,augment_data))
        if prob_trans_type == "lp":    
            nonlin = 'softmax'
            out_vec_size = num_classes_var
            cw_list = [None]
        elif prob_trans_type == "di":    
            nonlin = 'sigmoid'
            out_vec_size = num_classes_var
            cw_list = [None]
        elif prob_trans_type == "dc":    
            nonlin = 'softmax'
            out_vec_size = num_classes_var
            cw_list = [None]
        else:
            nonlin = 'sigmoid'
            out_vec_size = 1
            cw_list = [None]*len(trainY_noncat_list)
        if multi_task_tl == "multi_task" or multi_task_tl =="tr_learn":           
            for single_task_tup in single_inp_tasks_list:
                single_task_name, single_task_dict = single_task_tup
                if single_task_name == "topic":
                    single_task_dict['out_vec_size'] = single_task_dict['topic_vecs'].shape[1]
                    single_task_dict['nonlin'] = 'softmax'
                elif single_task_name == "kmeans":
                    single_task_dict['out_vec_size'] =single_task_dict['n_clusters']
                    single_task_dict['nonlin'] = 'softmax'
                single_inp_list.append([single_task_dict['out_vec_size'],single_task_dict['nonlin'],''])      
            for sep_task_tup in sep_inp_tasks_list:
                sep_task_name, sep_task_dict = sep_task_tup
                if sep_task_name == "sd":
                    sep_task_dict['out_vec_size'] =1
                    sep_task_dict['nonlin'] = 'sigmoid'
                if sep_task_name == "sepkmeans":
                    sep_task_dict['out_vec_size'] = sep_task_dict['n_clusters']
                    sep_task_dict['nonlin'] = 'softmax'
                if sep_task_name == "septopics":
                    sep_task_dict['out_vec_size'] =  sep_task_dict['num_classes_var']
                    sep_task_dict['nonlin'] = 'softmax'
                sep_inp_list.append([sep_task_dict['out_vec_size'],sep_task_dict['nonlin'],''])
        if class_imb_flag:
            if prob_trans_type == "di":
                cw_arr = np.empty([num_classes_var, 2])
                for i in range(num_classes_var):
                    cw_arr[i] = class_weight.compute_class_weight('balanced', [0,1], trainY_noncat_list[0][:, i])
                cw_list = [cw_arr]
                for ind,single_task_tup in enumerate(single_inp_tasks_list):
                    single_task_name, single_task_dict = single_task_tup
                    if  single_task_name == "kmeans":   
                        tr_uniq = np.arange(single_task_dict['n_clusters'])
                        single_task_dict['cw_aux'] = class_weight.compute_class_weight('balanced', tr_uniq, single_task_dict["kmeans_ids"])
                        single_inp_list[ind][2] = single_task_dict['cw_aux']
                for ind,sep_task_tup in enumerate(sep_inp_tasks_list):
                    sep_task_name, sep_task_dict = sep_task_tup
                    if  sep_task_name == "sepkmeans":   
                        tr_uniq = np.arange(sep_task_dict['n_clusters'])
                        sep_task_dict['cw_aux'] = class_weight.compute_class_weight('balanced', tr_uniq, sep_task_dict['trainY_noncat_list'][0])
                        sep_inp_list[ind][2] = sep_task_dict['cw_aux']
            elif prob_trans_type == "dc":
                cw_list = [weights_cat(trainY_noncat_list[0])]
            else:
                cw_list = []
                loss_func_list = []
                for trainY_noncat in trainY_noncat_list:
                    tr_uniq = np.arange(num_classes_var)
                    cw_arr = class_weight.compute_class_weight('balanced', tr_uniq, trainY_noncat)
                    cw_list.append(cw_arr)

        print("saving class imb for %s and %s; test mode = %s" % (class_imb_flag, prob_trans_type, test_mode))
        with open(filename, 'wb') as f:
            pickle.dump([nonlin, out_vec_size, cw_list,single_inp_list,sep_inp_list], f)

    if class_imb_flag:
        loss_func_list = []
        for cw_arr in cw_list:
            if prob_trans_type == "lp":
                loss_func_list.append(lp_categ_loss(cw_arr))
            elif prob_trans_type == "di":
                if use_conf_scores:
                    loss_func = multi_binary_loss_conf(cw_arr,out_vec_size)
                else:
                    if label_corr_setting != "None":                      
                        with open(save_fold_path + uncorr_c_pairs_filename, 'rb') as label_uncorr:
                            uncorrelated_c_pairs = pickle.load(label_uncorr)
                        if label_corr_setting == "norm_bin_pairs":
                            loss_func = corr_multi_binary_loss(cw_arr,uncorrelated_c_pairs,beta)
                        elif label_corr_setting == "unnorm_bin_pairs":
                            loss_func = unnorm_corr_multi_binary_loss(cw_arr,uncorrelated_c_pairs,beta)
                        elif label_corr_setting.startswith("l2_fuzzy_"):
                            loss_func = fuzzy_corr_multi_binary_loss(cw_arr,uncorrelated_c_pairs,beta)
                        elif label_corr_setting == "sep_norm_bin_pairs":
                            loss_func = sep_corr_multi_binary_loss(cw_arr,uncorrelated_c_pairs,beta)
                        elif label_corr_setting.startswith("l1_fuzzy_"):
                            loss_func = fuzzy_l1_corr_multi_binary_loss(cw_arr,uncorrelated_c_pairs,beta)
                    else:
                        loss_func = multi_binary_loss(cw_arr)
                if multi_task_tl == "multi_task" or multi_task_tl =="tr_learn":
                    losses, lossweights = get_lossfunc_weights(loss_func,classi_loss_weight,single_inp_tasks_list,sep_inp_tasks_list)
                    loss_func_list.append([losses,lossweights])
                else:
                    loss_func_list.append(loss_func)
            elif prob_trans_type == "dc":
                loss_func_list.append(multi_cat_w_loss(cw_arr))
            else:
                loss_func_list.append(br_binary_loss(cw_arr))
    else:
        if prob_trans_type == "lp":
            loss_func_list = ['categorical_crossentropy']
        elif prob_trans_type == "di":
            loss_func_list = ['binary_crossentropy']
        elif prob_trans_type == "dc":
            loss_func_list = [multi_cat_loss()]
        else:
            loss_func_list = ['binary_crossentropy']*len(trainY_noncat_list)

    return loss_func_list, nonlin, out_vec_size, cw_list

def transform_labels(data_trainY, prob_trans_type, test_mode, save_fold_path, filename, n_clusters,train_ratio):
    filename = "%slabel_info~%s~%s~%s~%s.pickle" % (save_fold_path, prob_trans_type, test_mode, filename,train_ratio)
    if os.path.isfile(filename):
        print("loading label info for %s; test mode = %s; filename = %s" % (prob_trans_type, test_mode,filename))
        with open(filename, 'rb') as f:
            trainY_list, trainY_noncat_list, num_classes_var, bac_map = pickle.load(f)
    else:
        if prob_trans_type == "lp":        
            lp_trainY, num_classes_var, bac_map, for_map = fit_trans_labels_powerset(data_trainY)
            print("num of LP classes: ", num_classes_var)
            trainY_noncat_list = [lp_trainY]
            trainY_list = [to_categorical(lp_trainY, num_classes=num_classes_var)]
        elif prob_trans_type == "di" or prob_trans_type == "dc":                      
            num_classes_var = NUM_CLASSES
            trainY_list = [trans_labels_multi_hot(data_trainY)]
            print("num of direct classes: ", num_classes_var)
            bac_map = None
            trainY_noncat_list = list(trainY_list)
        elif prob_trans_type == "br":    
            trainY_list = trans_labels_BR(data_trainY)
            num_classes_var = 2
            bac_map = None
            trainY_noncat_list = list(trainY_list)
        elif prob_trans_type == "binary":
            num_classes_var = 2
            trainY_list = trans_labels_bin_classi(data_trainY)
            bac_map = None
            trainY_noncat_list = list(trainY_list)
        elif prob_trans_type == "multi_class":
            num_classes_var = n_clusters
            bac_map = None
            trainY = trans_labels_multi_classi(data_trainY)
            trainY_noncat_list = [trainY]
            trainY_list = [to_categorical(trainY, num_classes=num_classes_var)]
        else:
            num_classes_var = np.array(data_trainY).shape[1]
            print (num_classes_var)
            bac_map = None
            trainY_noncat_list = None
            trainY_list = [np.array(data_trainY)]
            print (trainY_list)
        print("saving label info for %s; test mode = %s" % (prob_trans_type, test_mode))
        with open(filename, 'wb') as f:
            pickle.dump([trainY_list, trainY_noncat_list, num_classes_var, bac_map], f)
    return trainY_list, trainY_noncat_list, num_classes_var, bac_map
