import argparse
import os
import glob
# os.environ['OMP_NUM_THREADS'] = str(32)
import faiss
import numpy as np
import math
from util import load_tfrecords_and_index, read_id_dict, faiss_index
from multiprocessing import Pool
# import mkl
# mkl.set_num_threads(32)

# def index(corpus_embs, docids, save_path, quantize):

# 	dimension=corpus_embs.shape[1]
# 	cpu_index = faiss.IndexFlatIP(dimension)
# 	cpu_index = faiss.IndexIDMap(cpu_index)
# 	if quantize: # still try better way for balanced efficiency and effectiveness
# 		# ncentroids = 1000
# 		# code_size = dimension//4
# 		# cpu_index = faiss.IndexIVFPQ(cpu_index, dimension, ncentroids, code_size, 8)
# 		# cpu_index = faiss.IndexPQ(dimension, code_size, 8)
# 		cpu_index = faiss.index_factory(768, "OPQ128,IVF4096,PQ128", faiss.METRIC_INNER_PRODUCT)
# 		cpu_index = faiss.IndexIDMap(cpu_index)
# 		# cpu_index = faiss.GpuIndexScalarQuantizer(dimension, faiss.ScalarQuantizer.QT_16bit_direct, faiss.METRIC_INNER_PRODUCT)
# 		print("Train index...")
# 		cpu_index.train(corpus_embs)


# 	print("Indexing...")
# 	cpu_index.add_with_ids(corpus_embs, docids)
# 	faiss.write_index(cpu_index, save_path)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--index_path", type=str, required=True)
	parser.add_argument("--corpus_emb_path", type=str, required=True, help='The embedding file or the dir to save all the files with .tf')
	parser.add_argument("--doc_word_num", type=int, default=1, help='in case when using token embedding maxsim search instead of pooling embedding')
	parser.add_argument("--emb_dim", type=int, default=768)
	parser.add_argument("--passages_per_file", type=int, default=1000000, help='our default tf record include 1000,000 passages per file')
	parser.add_argument("--data_type", type=str, default='16', help='16 or 32 bit')
	parser.add_argument("--merge_index", action='store_true')
	parser.add_argument("--max_passage_each_index", type=int, default=50000000, help='Set a passage number limitation for index')
	parser.add_argument("--quantize", action='store_true')
	args = parser.parse_args()

	if not os.path.exists(args.index_path):
		os.mkdir(args.index_path)
	corpus_files=[]
	if os.path.isdir(args.corpus_emb_path):
		corpus_files = glob.glob(os.path.join(args.corpus_emb_path, '*.tf'))
	else:
		corpus_files = [FLAGS.corpus_path]



	if args.merge_index:
		print('Load %d tfrecord files...'%(len(corpus_files)))

		corpus_embs, docids=load_tfrecords_and_index(corpus_files, \
										   data_num=args.passages_per_file, \
										   word_num=args.doc_word_num, \
										   dim=args.emb_dim, data_type=args.data_type, index=False)

		corpus_embs = (corpus_embs.reshape((-1, args.emb_dim)))
		passage_num = corpus_embs.shape[0]

		index_file_num = int(math.ceil((passage_num)/float(args.max_passage_each_index)))
		if index_file_num > 1:
			for i in range(index_file_num):
				faiss_index(corpus_embs=corpus_embs[(i*args.max_passage_each_index):((i+1)*args.max_passage_each_index),:],
					  docids=docids[(i*args.max_passage_each_index):((i+1)*args.max_passage_each_index)],
					  save_path=os.path.join(args.index_path, 'index-' + str(i)),
					  quantize=args.quantize)
				print('index file:'+str(i))
		else:
			faiss_index(corpus_embs=corpus_embs,
				  docids=docids,
				  save_path=os.join.path(args.index_path, 'index'),
				  quantize=args.quantize)
	else:
		num_workers = len(corpus_files)
		pool = Pool(num_workers)

		for corpus_file in corpus_files:
			index_split = corpus_file.split('-')[-1]
			index_split = index_split.split('.')[0]

			pool.apply_async(load_tfrecords_and_index ,([corpus_file], args.passages_per_file, args.doc_word_num,
														args.emb_dim, args.data_type, True,
														os.path.join(args.index_path, 'index-' + index_split),
														args.quantize))

			# load_tfrecords_and_index([corpus_file], args.passages_per_file, args.doc_word_num,
			# 											args.emb_dim, args.data_type, True,
			# 											os.path.join(args.index_path, 'index-' + index_split),
			# 											args.quantize)
		pool.close()
		pool.join()


	print('finish')


if __name__ == "__main__":
	main()