import os
import pickle
os.environ['OMP_NUM_THREADS'] = str(16)
import faiss
# import mkl
# mkl.set_num_threads(16)
import numpy as np
import tensorflow.compat.v1 as tf
from numpy import linalg as LA
from progressbar import *
from collections import defaultdict
import glob

def read_pickle(filename):
	with open(filename, 'rb') as f:
		Distance, Index=pickle.load(f)
	return Distance, Index


def read_id_dict(path):
	if os.path.isdir(path):
		files = glob.glob(os.path.join(path, '*.id'))
	else:
		files = [path]

	idx_to_id = {}
	id_to_idx = {}
	for file in files:
		f = open(file, 'r')
		for i, line in enumerate(f):
			try:
				idx, Id =line.strip().split('\t')
				idx_to_id[int(idx)] = Id
				id_to_idx[Id] = int(idx)
			except:
				print(line+' has no id')
	return idx_to_id, id_to_idx

def write_result(qidxs, Index, Score, file, idx_to_qid, idx_to_docid, topk=None, run_name='Faiss'):
	print('write results...')
	with open(file, 'w') as fout:
		for i, qidx in enumerate(qidxs):
			try:
				qid = idx_to_qid[qidx]
			except:
				qid = qidx
			if topk==None:
				docidxs=Index[i]
				scores=Score[i]
				for rank, docidx in enumerate(docidxs):
					try:
						docid = idx_to_docid[docidx]
					except:
						docid = docidx
					fout.write('{} Q0 {} {} {} {}\n'.format(qid, docid, rank + 1, scores[rank], run_name))
			else:
				try:
					hit=min(topk, len(Index[i]))
				except:
					print('debug')

				docidxs=Index[i]
				scores=Score[i]
				for rank, docidx in enumerate(docidxs[:hit]):
					try:
						docid = idx_to_docid[docidx]
					except:
						docid = docidx
					fout.write('{} Q0 {} {} {} {}\n'.format(qid, docid, rank + 1, scores[rank], run_name))
def load_term_weight_tfrecords(srcfiles, dim, data_type='16', index=False, batch=1):
	def _parse_function(example_proto):
		features = {'term_weight': tf.FixedLenFeature([],tf.string) , #tf.FixedLenSequenceFeature([],tf.string, allow_missing=True),
					'docid': tf.FixedLenFeature([],tf.int64)}
		parsed_features = tf.parse_single_example(example_proto, features)
		if data_type=='16':
			corpus = tf.decode_raw(parsed_features['term_weight'], tf.float16)
		elif data_type=='32':
			corpus = tf.decode_raw(parsed_features['term_weight'], tf.float32)
		docid = tf.cast(parsed_features['docid'], tf.int32)
		return corpus, docid
	print('Read embeddings...')
	# widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),
	# 	' ', ETA(), ' ', FileTransferSpeed()]
	# pbar = ProgressBar(widgets=widgets, maxval=10*data_num*len(srcfiles)).start()
	with tf.Session() as sess:
		docids=[]
		term_weights=[]
		#assign memory in advance so that we can save memory without concatenate

		# if (data_type=='16'): # Faiss now only support index array with float32
		# 	corpus_embs = np.zeros((word_num*data_num*len(srcfiles) , dim), dtype=np.float16)
		# elif data_type=='32':
		# 	corpus_embs = np.zeros((word_num*data_num*len(srcfiles) , dim), dtype=np.float32)
		# else:
		# 	raise Exception('Please assign datatype 16 or 32 bits')
		counter = 0
		i = 0
		for srcfile in srcfiles:
			try:
				dataset = tf.data.TFRecordDataset(srcfile) # load tfrecord file
			except:
				print('Cannot find data')
				continue
			dataset = dataset.map(_parse_function) # parse data into tensor
			dataset = dataset.repeat(1)
			dataset = dataset.batch(batch)
			iterator = dataset.make_one_shot_iterator()
			next_data = iterator.get_next()

			while True:
				try:
					corpus_emb, docid = sess.run(next_data)
					corpus_emb = corpus_emb.reshape(-1)
					sent_num = corpus_emb.shape[0]
					docids.append(docid)
					term_weights.append(corpus_emb)
					counter+=sent_num
					# pbar.update(10 * i + 1)
					# i+=sent_num

				except tf.errors.OutOfRangeError:
					break
	return term_weights, docids

def faiss_index(corpus_embs, docids, save_path, quantize):

	dimension=corpus_embs.shape[1]
	cpu_index = faiss.IndexFlatIP(dimension)
	cpu_index = faiss.IndexIDMap(cpu_index)
	if quantize: # still try better way for balanced efficiency and effectiveness
		# ncentroids = 1000
		# code_size = dimension//4
		# cpu_index = faiss.IndexIVFPQ(cpu_index, dimension, ncentroids, code_size, 8)
		# cpu_index = faiss.IndexPQ(dimension, code_size, 8)
		cpu_index = faiss.index_factory(768, "OPQ128,IVF4096,PQ128", faiss.METRIC_INNER_PRODUCT)
		cpu_index = faiss.IndexIDMap(cpu_index)
		# cpu_index = faiss.GpuIndexScalarQuantizer(dimension, faiss.ScalarQuantizer.QT_16bit_direct, faiss.METRIC_INNER_PRODUCT)
		print("Train index...")
		cpu_index.train(corpus_embs)


	print("Indexing {}...".format(save_path))
	cpu_index.add_with_ids(corpus_embs, docids)
	faiss.write_index(cpu_index, save_path)


def load_tfrecords_and_index(srcfiles, data_num, word_num, dim, data_type, index=False, save_path=None, quantize=None, batch=1000):
	def _parse_function(example_proto):
		features = {'doc_emb': tf.FixedLenFeature([],tf.string) , #tf.FixedLenSequenceFeature([],tf.string, allow_missing=True),
					'docid': tf.FixedLenFeature([],tf.int64)}
		parsed_features = tf.parse_single_example(example_proto, features)
		if data_type=='16':
			corpus = tf.decode_raw(parsed_features['doc_emb'], tf.float16)
		elif data_type=='32':
			corpus = tf.decode_raw(parsed_features['doc_emb'], tf.float32)
		docid = tf.cast(parsed_features['docid'], tf.int32)
		return corpus, docid
	print('Read embeddings...')
	widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),
		' ', ETA(), ' ', FileTransferSpeed()]
	pbar = ProgressBar(widgets=widgets, maxval=10*data_num*len(srcfiles)).start()
	with tf.Session() as sess:
		docids=[]
		#assign memory in advance so that we can save memory without concatenate

		if (data_type=='16'): # Faiss now only support index array with float32
			corpus_embs = np.zeros((word_num*data_num*len(srcfiles) , dim), dtype=np.float16)
		elif data_type=='32':
			corpus_embs = np.zeros((word_num*data_num*len(srcfiles) , dim), dtype=np.float32)
		# else:
		# 	raise Exception('Please assign datatype 16 or 32 bits')
		counter = 0
		i = 0
		for srcfile in srcfiles:
			try:
				dataset = tf.data.TFRecordDataset(srcfile) # load tfrecord file
			except:
				print('Cannot find data')
				continue
			dataset = dataset.map(_parse_function) # parse data into tensor
			dataset = dataset.repeat(1)
			dataset = dataset.batch(batch)
			iterator = dataset.make_one_shot_iterator()
			next_data = iterator.get_next()

			while True:
				try:
					corpus_emb, docid = sess.run(next_data)
					corpus_emb = corpus_emb.reshape(-1, dim)

					sent_num = corpus_emb.shape[0]
					corpus_embs[counter:(counter+sent_num)] = corpus_emb

					docids+=docid.tolist()
					counter+=sent_num
					pbar.update(10 * i + 1)
					i+=sent_num

				except tf.errors.OutOfRangeError:
					break

		docids = np.array(docids).reshape(-1)
		corpus_embs = (corpus_embs[:len(docids)]).astype(np.float32)
	if index:
		faiss_index(corpus_embs, docids, save_path, quantize)
	else:
		return corpus_embs, docids


def normalize(embeddings):
	return (embeddings.T/LA.norm(embeddings,axis=-1)).T