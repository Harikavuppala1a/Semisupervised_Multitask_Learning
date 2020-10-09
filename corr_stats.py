import sys
import numpy as np
from loadPreProc import *

thresh = float(sys.argv[2])

def get_filename(filename):
    return filename.split('.')[0]

conf_dict_list, conf_dict_com = load_config(sys.argv[1])
data_dict = load_data(conf_dict_com["filename"], conf_dict_com["data_folder_name"], conf_dict_com["save_folder_name"], conf_dict_com['TEST_RATIO'], conf_dict_com['VALID_RATIO'], conf_dict_com['RANDOM_STATE'], conf_dict_com['MAX_WORDS_SENT'], conf_dict_com["test_mode"],conf_dict_com['train_ratio'])
prime_filename = get_filename(conf_dict_com['filename'])
data_trainY = data_dict['lab'][:data_dict['train_en_ind']]

co_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
freqs = np.zeros(NUM_CLASSES)
for t_cats in data_trainY:
	for i in range(len(t_cats)):
		freqs[t_cats[i]] += 1
		for j in range(i+1, len(t_cats)):
			if t_cats[i] < t_cats[j]:
				co_mat[t_cats[i], t_cats[j]] += 1
			else:
				co_mat[t_cats[j], t_cats[i]] += 1

for i in range(1, NUM_CLASSES):
	for j in range(0, i):
		co_mat[i,j] = co_mat[j,i]

#P(j|i) = co_mat[i,j]
for i in range(NUM_CLASSES):
	co_mat[i,:] = co_mat[i,:]/freqs[i]

#uncorrelated_c_pairs = S
uncorrelated_c_pairs = []
for i in range(NUM_CLASSES):
	for j in range(NUM_CLASSES):
		if (co_mat[i,j] <= thresh) and (i != j):
			uncorrelated_c_pairs.append((i,j,co_mat[i,j]))
	co_mat[i,i] = 1

filename = "%scorr_fuzzy_%s.pickle" % (conf_dict_com["save_folder_name"], thresh)
with open(filename, 'wb') as f:
    pickle.dump(uncorrelated_c_pairs, f)

filename = "%snorm_corr_mat.pickle" % (conf_dict_com["save_folder_name"])
with open(filename, 'wb') as f:
    pickle.dump(co_mat, f)
