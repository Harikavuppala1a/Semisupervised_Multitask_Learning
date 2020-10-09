import json
import csv
from os import listdir
from os.path import isfile, join
from loadPreProc import *
from evalMeasures import *
import numpy as np
import re
from nltk import sent_tokenize
import pickle

ours_fname = "../bestmodel.txt"
base_fname = "../baseline.txt"

train_coverage_fname = "../train_coverage.txt"
samples_fname = "../selected_samples.txt"

classwise_perf_fname = "../classwise_performance.csv"

min_samples = 5


def build_post_dict():
    filename = 'saved/cl_orig_dict.pickle' 
    if os.path.isfile(filename):
        print("loading cl_orig_dict")
        with open(filename, 'rb') as f:
            cl_orig_dict = pickle.load(f)
    else:
        task_filename = 'data/data_trans.csv'
        r_anum = re.compile(r'([^\sa-z0-9.(?)!])+')
        r_white = re.compile(r'[\s.(?)!]+')
        max_words_sent = 35
        cl_orig_dict = {}
        with open(task_filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile, delimiter = '\t')
            for row in reader:
                post = str(row['post'])

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
                cl_orig_dict[' '.join(se_list)] = post
        print("saving cl_orig_dict")
        with open(filename, 'wb') as f:
            pickle.dump(cl_orig_dict, f)

    return cl_orig_dict

def write_error_ana(dict_list, num_runs, min_num_labs, max_mum_labs, pred_freq_list, f_summary):
    metr_dict = init_metr_dict()
    actu_freqs = {num_labs: 0 for num_labs in range(min_num_labs, max_mum_labs+1)}
    pred_freqs = {num_labs: 0 for num_labs in range(min_num_labs, max_mum_labs+1)}
    for num_labs in range(min_num_labs, max_mum_labs+1):
        for n_r in range(num_runs): 
            actu_freqs[num_labs] += len(dict_list[n_r][num_labs]['true_vals'])
            pred_freqs[num_labs] += pred_freq_list[n_r][num_labs]
        actu_freqs[num_labs] = round(actu_freqs[num_labs]/num_runs)
        pred_freqs[num_labs] = round(pred_freqs[num_labs]/num_runs)

        if len(dict_list[0][num_labs]['true_vals']) < min_samples:
            continue        
        print("number of labels per post: %s" % num_labs)
        f_summary.write("number of labels per post: %s\n" % num_labs)
        for n_r in range(num_runs): 
            metr_dict = calc_metrics_print(dict_list[n_r][num_labs]['pred_vals'], dict_list[n_r][num_labs]['true_vals'], metr_dict)
        metr_dict = aggregate_metr(metr_dict, num_runs)

        write_results(metr_dict, f_summary)
        f_summary.write("----------------------\n")                                                           
        print("----------------------")                                                           

    f_summary.write("# of labels per post\tfrequency wrt the true labels\tfrequency wrt the predicted labels\t\n")  
    for num_labs in range(min_num_labs, max_mum_labs+1):
        f_summary.write("%d\t%d\t%d\t\n" % (num_labs, actu_freqs[num_labs], pred_freqs[num_labs]))                                                           

def error_ana(ours_fname, base_fname, num_runs):
    min_num_labs = 1
    max_mum_labs = 10
    header_strings = ["Best proposed method", "Best baseline"]
    f_summary = open("../error_ana.txt", 'w')
    for file_ind, filename in enumerate([ours_fname, base_fname]):
        dict_list = []
        pred_freq_list = []
        for n_r in range(num_runs):
            num_lab_dict = {}
            num_lab_freq_dict = {}
            for i in range(min_num_labs, max_mum_labs+1):
                num_lab_dict[i] = {'pred_vals': [], 'true_vals': []}
                num_lab_freq_dict[i] = 0
            dict_list.append(num_lab_dict)
            pred_freq_list.append(num_lab_freq_dict)

        for n_r in range(num_runs):
            with open(filename,'r') as f:
                reader = csv.DictReader(f, delimiter = '\t')
                rows = list(reader)
            for i in range(len(rows)):

                t_cats = [int(st.strip()) for st in rows[i]['actu cats'].split(',')]
                p_cats = [int(st.strip()) for st in rows[i]['pred cats'].split(',')]
                dict_list[n_r][len(t_cats)]['true_vals'].append(t_cats)
                dict_list[n_r][len(t_cats)]['pred_vals'].append(p_cats)
                pred_freq_list[n_r][len(p_cats)] += 1
                # count = count + 1
        f_summary.write("%s\n\n" % header_strings[file_ind])
        print("%s\n" % header_strings[file_ind])
        write_error_ana(dict_list, num_runs, min_num_labs, max_mum_labs, pred_freq_list, f_summary)
        f_summary.write("****************************\n")
        print("****************************")

    f_summary.close()

error_ana(ours_fname, base_fname, 1)

def get_samples(ours_fname, base_fname,train_coverage_fname, min_num_labels_thresh):
    orig_dict = build_post_dict()
    with open(ours_fname,'r') as fprop:
        reader = csv.DictReader(fprop, delimiter = '\t')
        prop_rows = list(reader)
    with open(base_fname,'r') as fbase:
        reader = csv.DictReader(fbase, delimiter = '\t')
        base_rows = list(reader)
    with open(train_coverage_fname,'r') as tc:
        reader = csv.DictReader(tc, delimiter = '\t')
        tc_rows = list(reader)

    with open("../selected_samples.txt", 'w') as w_samples:
        w_samples.write("post\ttrue labels\tmean train cov\tproposed predicted labels\tbaseline predicted labels\tjaccard for proposed\tjaccard for baseline\n")
        for i in range(len(prop_rows)):
            t_cats_int = [int(st.strip()) for st in prop_rows[i]['actu cats'].split(',')]
            t_cats = [FOR_LMAP[c_id] for c_id in t_cats_int]
            prop_p_cats = [FOR_LMAP[int(st.strip())] for st in prop_rows[i]['pred cats'].split(',')]
            base_p_cats = [FOR_LMAP[int(st.strip())] for st in base_rows[i]['pred cats'].split(',')]
            prop_jaccard = prop_rows[i]['Jaccard']
            base_jaccard = base_rows[i]['Jaccard']
            train_coverage = [float(tc_rows[c_id]['train cov']) for c_id in t_cats_int]
            if len(t_cats) >= min_num_labels_thresh:
                prop_rows[i]['post'] = prop_rows[i]['post'].replace('._', " ")
                w_samples.write("%s\t%s\t%.3f\t%s\t%s\t%s\t%s\n" % (orig_dict[prop_rows[i]['post']], ', '.join(t_cats), np.mean(train_coverage), ', '.join(prop_p_cats), ', '.join(base_p_cats),prop_jaccard,base_jaccard))
get_samples(ours_fname, base_fname,train_coverage_fname, 3)

def get_count(classwise_perf_fname):
    with open(classwise_perf_fname,'r') as cw:
        reader = csv.DictReader(cw)
        count = 0
        for row in reader:
            if row['best_proposed_F score'] > row["best_baseline_F score"]:
                count = count + 1
        print("Number of labels for which ours outperforms: %s; percent =" %.2f % (count, count*100/NUM_CLASSES))
get_count(classwise_perf_fname)
