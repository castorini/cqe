import argparse
import pickle
import glob
import os
import numpy as np
from util import read_id_dict, read_pickle, write_result
from progressbar import *


def dedupe_index(Index, Score):
	NewIndex = []
	NewScore = []
	max_index_num = 0
	min_index_num = 10000000
	doc_num=0
	sort_index=np.argsort(Score)[:,::-1]


	for i, index in enumerate(Index):
		sort_id = index[sort_index[i].tolist()]
		sort_score = Score[i][sort_index[i].tolist()]

		uniq_indices=np.sort(np.unique(sort_id,return_index=True)[1]).tolist()#the index for dedupe

		max_index_num = max(max_index_num, len(uniq_indices))
		min_index_num  = min(min_index_num, len(uniq_indices))
		doc_num+=len(uniq_indices)
		NewIndex.append(sort_id[uniq_indices].tolist())
		NewScore.append(sort_score[uniq_indices].tolist())
	print("Maximum unique doc id is %d after dedupe"%(max_index_num))
	print("Minimum unique doc id is %d after dedupe"%(min_index_num))
	print("Average unique doc id is %d after dedupe"%(doc_num/Index.shape[0]))
	return NewIndex, NewScore, max_index_num



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--intermediate_path", type=str)
	parser.add_argument("--output", type=str)
	parser.add_argument("--docs_per_file", type=int, default=1000000)
	parser.add_argument("--doc_word_num", type=int, default=1)
	parser.add_argument("--num_files", type=int, default=10)
	parser.add_argument("--query_word_num", type=int, default=1)
	parser.add_argument("--topk", type=int, default=1000)
	parser.add_argument("--emb_dim", type=int, default=768)
	parser.add_argument("--data_type", type=str, default=16)
	parser.add_argument("--id_to_doc_path", type=str, default=None)
	parser.add_argument("--id_to_query_path", type=str, default=None)
	parser.add_argument("--corpus_type", type=str, default='passage')
	args = parser.parse_args()

	idx_to_docid, docid_to_idx = read_id_dict(args.id_to_doc_path)
	idx_to_qid, _ = read_id_dict(args.id_to_query_path)
	qidxs = idx_to_qid.values()

	if args.corpus_type=='doc':
		idx_to_uniq_docid = {}
		for idx in idx_to_docid.keys():
			docid = idx_to_docid[idx]
			uniq_docid = docid_to_idx[docid]
			idx_to_uniq_docid[idx] = uniq_docid

	Score=None
	Index=None
	for filename in glob.glob(os.path.join(args.intermediate_path, '*')):
		S, I=read_pickle(filename)
		try:
			Score = np.concatenate([Score, S], axis=1)
			Index= np.concatenate([Index, I], axis=1)
		except:
			Score=S
			Index=I

	if args.corpus_type=='doc':
		new_index=[]
		for index_row in Index:
			temp = []
			for index in index_row:
				temp.append(idx_to_uniq_docid[index])
			new_index.append(temp)
		Index = np.array(new_index)

	print("Dedupe...")

	Index, Score, max_index_num=dedupe_index(Index, Score)

	write_result(qidxs, Index, Score, args.output, idx_to_qid, idx_to_docid, args.topk)

if __name__ == "__main__":
	main()