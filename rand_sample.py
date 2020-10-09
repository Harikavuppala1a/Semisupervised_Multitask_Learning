import sys
import csv
import operator
import re
import random
from nltk import sent_tokenize
csv.field_size_limit(sys.maxsize)

neg_data_filename = "data/blogtext_selected.csv"
pos_data_filename = "data/unlab_minus_lab_shortest_n.txt"
opfilename = "data/sexismDet.txt"
sampleSize = 20000
max_words_sent = 35
max_sent = 16
max_post_length =198

r_anum = re.compile(r'([^\sa-z0-9.(?)!])+')
r_white = re.compile(r'[\s.(?)!]+')

def get_data (filename):
	origList =[]
	with open(filename) as csvfile:
		reader = csv.reader(csvfile, delimiter='\t')
		for row in reader:
			post = str(row[0])
			row_clean = r_white.sub(' ', r_anum.sub('', post.lower())).strip()
			if len(row_clean.split(' ')) <= max_post_length:
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
				if len(se_list) <= max_sent:
						origList.append(row)
	return origList

origList_pos= get_data(pos_data_filename)
origList_neg = get_data(neg_data_filename)

sList_pos = random.sample(origList_pos, sampleSize)
sList_neg = random.sample(origList_neg, sampleSize)
final_list = sList_pos + sList_neg

with open(opfilename, 'w') as opfile:
	wr = csv.writer(opfile, delimiter = '\t')
	header = ['post','label']
	wr.writerow(header)
	for num,entry in enumerate(final_list):
		if num < sampleSize:		
			row = [entry[0],1]
		else:
			row = [entry[0],0]
		wr.writerow(row)