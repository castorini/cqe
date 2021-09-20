from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import numpy as np
from numpy import linalg as LA
import tensorflow as tf
from model_fn import create_bert, create_model, model_fn_builder, input_fn_builder
import metrics
import modeling
from tqdm import tqdm


flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_bool(
	"use_tpu",False, 'whether to use TPU')
flags.DEFINE_bool(
	"do_train",False, 'Train')
flags.DEFINE_bool(
	"do_eval",False, 'Eval')
flags.DEFINE_bool(
	"do_output",False, 'Output embedding')
flags.DEFINE_string(
	"tpu_address",
	None, None)
flags.DEFINE_string(
	"kd_source",
	'colbert', 'colbert or label')
flags.DEFINE_string(
	"loss",
	'kl', 'kl or mse')
flags.DEFINE_string(
	"train_model",
	'student', 'student or teacher')
flags.DEFINE_string(
	"eval_model",
	'student', 'student or teacher')
flags.DEFINE_string(
	"bert_pretrained_dir",
	None, None)
flags.DEFINE_string(
	"output_dir",
	None, None)
flags.DEFINE_string(
	"data_dir",
	None, None)
flags.DEFINE_string(
	'train_file',
	'dataset_train_tower.tf', None)
flags.DEFINE_string(
	'eval_file',
	'dataset_dev_tower.tf', None)
flags.DEFINE_string(
	'eval_id_map',
	'query_doc_ids_dev.txt', None)
flags.DEFINE_string(
	'eval_output_file',
	'cast_rerank.tsv', None)
flags.DEFINE_string(
	'init_checkpoint',
	None, None)
flags.DEFINE_string(
	'eval_checkpoint',None, None)
flags.DEFINE_string(
	'embedding_file',None, None)
flags.DEFINE_integer(
	"train_batch_size", 96, None)
flags.DEFINE_integer(
	"eval_batch_size", 40, None)
flags.DEFINE_float(
	"learning_rate", 7e-6, None)
flags.DEFINE_integer(
	"num_train_steps", 100000, None)
flags.DEFINE_integer(
	"num_warmup_steps", 10000, None, None)
flags.DEFINE_integer(
	"max_query_length", 136, None)
flags.DEFINE_integer(
	"max_doc_length", 154, None)
flags.DEFINE_integer(
	"colbert_dim", 128, None)
flags.DEFINE_integer(
	"dotbert_dim", 768, None)
flags.DEFINE_integer(
	"doc_type", 1, '0: Query, 1: Doc')
flags.DEFINE_integer(
	"candidates_per_query", 1000, None)
flags.DEFINE_integer(
	"save_sample_num", 1000000, '# of output embedding in a file')
flags.DEFINE_integer(
	"save_checkpoints_steps", 10000, '# of steps for saving checkpoint')
flags.DEFINE_integer(
	"max_eval_examples", None, None)
flags.DEFINE_integer(
	"num_tpu_cores", 8, None)
# Configures
USE_TPU = FLAGS.use_tpu
TPU_ADDRESS = FLAGS.tpu_address
if USE_TPU:
	if TPU_ADDRESS==None:
		raise ValueError('No tpu address')
DO_TRAIN = FLAGS.do_train #Whether to run training.
DO_EVAL = FLAGS.do_eval # Whether to run evaluation.
DO_OUTPUT = FLAGS.do_output # Whether to output embeddings.
BERT_PRETRAINED_DIR = FLAGS.bert_pretrained_dir

TRAIN_BATCH_SIZE = FLAGS.train_batch_size
EVAL_BATCH_SIZE = FLAGS.eval_batch_size
Save_SAMPLE_NUM =FLAGS.save_sample_num
CANDIDATES_PER_QUERY = FLAGS.candidates_per_query  # Number of docs per query in the dev and eval files.
MAX_EVAL_EXAMPLES = FLAGS.max_eval_examples  # Maximum number of examples to be evaluated.
SAVE_CHECKPOINTS_STEPS = FLAGS.save_checkpoints_steps
KD_SOURCE = FLAGS.kd_source
LOSS = FLAGS.loss
if KD_SOURCE not in ['colbert', 'label', 'combine']:
	raise ValueError('Use kd_source: colbert, label, combine')
if LOSS not in ['kl', 'mse']:
	raise ValueError('Use loss: kl, mse')
TRAIN_MODEL = FLAGS.train_model
EVAL_MODEL = FLAGS.eval_model
if EVAL_MODEL not in ['student', 'teacher']:
	raise ValueError('Use eval_model: student, teacher')
ITERATIONS_PER_LOOP = 100
NUM_TPU_CORES = FLAGS.num_tpu_cores
# Hyperparameters
LEARNING_RATE = FLAGS.learning_rate #7e-6
NUM_TRAIN_STEPS = FLAGS.num_train_steps
NUM_WARMUP_STEPS = FLAGS.num_warmup_steps
MAX_Query_LENGTH = FLAGS.max_query_length #136
MAX_Doc_LENGTH = FLAGS.max_doc_length #154
COLBERT_DIM = FLAGS.colbert_dim
DOTBERT_DIM = FLAGS.dotbert_dim
# Input Files
OUTPUT_DIR = FLAGS.output_dir
DATA_DIR = FLAGS.data_dir
TRAIN_FILE = os.path.join(DATA_DIR, FLAGS.train_file)
EVAL_FILE = os.path.join(DATA_DIR, FLAGS.eval_file)
EVAL_ID_MAP = os.path.join(DATA_DIR, FLAGS.eval_id_map)
EVAL_OUTPUT_FILE = os.path.join(OUTPUT_DIR, FLAGS.eval_output_file)
BERT_CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
INIT_CHECKPOINT = FLAGS.init_checkpoint
EVAL_CHECKPOINT = FLAGS.eval_checkpoint
MSMARCO_OUTPUT = True  # Write the predictions to a MS-MARCO-formatted file.
DOC_TYPE = FLAGS.doc_type
if INIT_CHECKPOINT==None:
	INIT_CHECKPOINT = EVAL_CHECKPOINT


if DO_EVAL:
	METRICS_MAP = ['MAP', 'RPrec', 'NDCG', 'MRR', 'MRR@10']

if DO_OUTPUT:
	if FLAGS.embedding_file==None:
		raise ValueError('Input embedding file for output')
	else:
		tf.gfile.MkDir(OUTPUT_DIR)
		EMBEDDING_FILE = FLAGS.embedding_file
		split = EMBEDDING_FILE.split('-')[-1]
		output_emb_file = os.path.join(OUTPUT_DIR, "embeddings-"+ str(split) + ".tf")



print('***** Model output directory: {} *****'.format(OUTPUT_DIR))


def prepare():
	tf.logging.set_verbosity(tf.logging.INFO)

	if not DO_TRAIN and not DO_EVAL and not DO_OUTPUT:
		raise ValueError("At least one of `DO_TRAIN` or `DO_EVAL`  or `DO_OUTPUT` must be True.")

	bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG_FILE)
	if MAX_Doc_LENGTH > bert_config.max_position_embeddings:
		raise ValueError(
			"Cannot use sequence length %d because the BERT model "
			"was only trained up to sequence length %d" %
			(MAX_Doc_LENGTH, bert_config.max_position_embeddings))


	tpu_cluster_resolver = None
	if USE_TPU:
		tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
			TPU_ADDRESS)

	is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
	run_config = tf.contrib.tpu.RunConfig(
				cluster=tpu_cluster_resolver,
				model_dir=OUTPUT_DIR,
				save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
				tpu_config=tf.contrib.tpu.TPUConfig(
					iterations_per_loop=ITERATIONS_PER_LOOP,
					num_shards=NUM_TPU_CORES,
					per_host_input_for_training=is_per_host)
				)

	model_fn = model_fn_builder(
				bert_config=bert_config,
				init_checkpoint=INIT_CHECKPOINT,
				learning_rate=LEARNING_RATE,
				num_train_steps=NUM_TRAIN_STEPS,
				num_warmup_steps=NUM_WARMUP_STEPS,
				use_tpu=USE_TPU,
				use_one_hot_embeddings=USE_TPU,
				colbert_dim=COLBERT_DIM, dotbert_dim=DOTBERT_DIM,
				max_q_len=MAX_Query_LENGTH, max_p_len=MAX_Doc_LENGTH, doc_type=DOC_TYPE,
				loss= LOSS, kd_source=KD_SOURCE, train_model=TRAIN_MODEL, eval_model=EVAL_MODEL,
				is_eval=DO_EVAL,
				is_output=DO_OUTPUT)

	# If TPU is not available, this will fall back to normal Estimator on CPU
	# or GPU.
	estimator = tf.contrib.tpu.TPUEstimator(
				use_tpu=USE_TPU,
				model_fn=model_fn,
				config=run_config,
				train_batch_size=TRAIN_BATCH_SIZE,
				eval_batch_size=EVAL_BATCH_SIZE,
				predict_batch_size=EVAL_BATCH_SIZE)
	return model_fn, estimator


def main(_):
	model_fn, estimator = prepare()
	if DO_TRAIN:
		tf.logging.info("***** Running training *****")
		tf.logging.info("  Batch size = %d", TRAIN_BATCH_SIZE)
		tf.logging.info("  Num steps = %d", NUM_TRAIN_STEPS)

		train_input_fn = input_fn_builder(
							dataset_path=TRAIN_FILE,
							max_q_len=MAX_Query_LENGTH, max_p_len=MAX_Doc_LENGTH, doc_type=DOC_TYPE,
							is_eval=False, is_output=False, is_training=True)

		estimator.train(input_fn=train_input_fn,
						max_steps=NUM_TRAIN_STEPS)
		tf.logging.info("Done Training!")
	if DO_EVAL:
		tf.logging.info("***** Running evaluation *****")
		tf.logging.info("  Batch size = %d", EVAL_BATCH_SIZE)
		max_eval_examples = None
		if MAX_EVAL_EXAMPLES:
			max_eval_examples = MAX_EVAL_EXAMPLES * CANDIDATES_PER_QUERY

		eval_input_fn = input_fn_builder(
						dataset_path= EVAL_FILE,
						max_q_len=MAX_Query_LENGTH, max_p_len=MAX_Doc_LENGTH, doc_type=DOC_TYPE,
						is_eval=True, is_output=False, is_training=False,
						max_eval_examples=max_eval_examples)

		if MSMARCO_OUTPUT:
			msmarco_file = tf.gfile.Open(
				EVAL_OUTPUT_FILE, "w")

		query_docids_map = []
		with tf.gfile.Open(EVAL_ID_MAP) as ref_file:
			for line in ref_file:
				query_docids_map.append(line.strip().split("\t"))
		# ***IMPORTANT NOTE***
		# The logging output produced by the feed queues during evaluation is very
		# large (~14M lines for the dev set), which causes the tab to crash if you
		# don't have enough memory on your local machine. We suppress this
		# frequent logging by setting the verbosity to WARN during the evaluation
		# phase.
		tf.logging.set_verbosity(tf.logging.WARN)

		result = estimator.predict(input_fn=eval_input_fn,
								yield_single_examples=True, checkpoint_path=EVAL_CHECKPOINT) #
		start_time = time.time()
		results = []
		all_metrics = np.zeros(len(METRICS_MAP))
		example_idx = 0
		total_count = 0
		score_list = []
		qid_list =[]
		for item in result:
			results.append((item["log_probs"], item["label_ids"]))
			#print(debug)
			if len(results) == CANDIDATES_PER_QUERY:

				log_probs, labels = zip(*results)
				log_probs = np.stack(log_probs) #.reshape(-1, 1)
				labels = np.stack(labels)

				scores = log_probs #[:, 0]
				pred_docs = scores.argsort()[::-1]
				#pred_docs = np.array([i for i in range(1000)])
				gt = set(list(np.where(labels > 0)[0]))
				# max_score = np.max(scores)

				all_metrics += metrics.metrics(
					gt=gt, pred=pred_docs, metrics_map=METRICS_MAP)

				# eval_scores = metrics.metrics(
				#     gt=gt, pred=pred_docs, metrics_map=METRICS_MAP)
				# score_list.append( (eval_scores[0],max_score) )
				# all_metrics += eval_scores

				if MSMARCO_OUTPUT:
					start_idx = example_idx * CANDIDATES_PER_QUERY
					end_idx = (example_idx + 1) * CANDIDATES_PER_QUERY
					query_ids, doc_ids = zip(*query_docids_map[start_idx:end_idx])
					assert len(set(query_ids)) == 1, "Query ids must be all the same."
					query_id = query_ids[0]
					qid_list.append(query_id)
					rank = 1
					for doc_idx in pred_docs:
						doc_id = doc_ids[doc_idx]
						# Skip fake docs, as they are only used to ensure that each query
						# has 1000 docs.
						if doc_id != "00000000":
							msmarco_file.write(
							"\t".join((query_id, doc_id, str(rank), str(scores[doc_idx]))) + "\n")
							rank += 1

				example_idx += 1
				results = []

			total_count += 1

			if total_count % 10000 == 0:
				tf.logging.warn("Read {} examples in {} secs. Metrics so far:".format(
								total_count, int(time.time() - start_time)))
				tf.logging.warn("  ".join(METRICS_MAP))
				tf.logging.warn(all_metrics / example_idx)

		tf.logging.set_verbosity(tf.logging.INFO)

		if MSMARCO_OUTPUT:
			msmarco_file.close()

		all_metrics /= example_idx

		tf.logging.info("Eval {}:".format(EVAL_FILE))
		tf.logging.info("  ".join(METRICS_MAP))
		tf.logging.info(all_metrics)

	if DO_OUTPUT:
		# for i, set_name in enumerate(File_list): #, "queries.dev0", "msmarco0", "msmarco1"
			# DOC_ID = DOC_ID_list[i]
			# NUM_TPU_COREs = NUM_TPU_COREs_list[i]
			# EVAL_BATCH_SIZE = EVAL_BATCH_SIZE_list[i]

		tf.logging.info("***** Output Embedding:"+ EMBEDDING_FILE + "*****")
		tf.logging.info("  Batch size = %d", EVAL_BATCH_SIZE)
		max_eval_examples = None
		if MAX_EVAL_EXAMPLES:
			max_eval_examples = MAX_EVAL_EXAMPLES * CANDIDATES_PER_QUERY



		eval_input_fn = input_fn_builder(
			dataset_path=os.path.join(DATA_DIR, EMBEDDING_FILE + '.tf'),
			max_q_len=MAX_Query_LENGTH, max_p_len=MAX_Doc_LENGTH, doc_type=DOC_TYPE,
			is_eval=False, is_output=True, is_training=False,
			max_eval_examples=max_eval_examples)

		counter=0
		if MSMARCO_OUTPUT:
			writer = tf.python_io.TFRecordWriter(
				os.path.join(output_emb_file))


		# ***IMPORTANT NOTE***
		# The logging output produced by the feed queues during evaluation is very
		# large (~14M lines for the dev set), which causes the tab to crash if you
		# don't have enough memory on your local machine. We suppress this
		# frequent logging by setting the verbosity to WARN during the evaluation
		# phase.
		tf.logging.set_verbosity(tf.logging.WARN)

		result = estimator.predict(input_fn=eval_input_fn,
		                            yield_single_examples=True, checkpoint_path=EVAL_CHECKPOINT)

		start_time =0 
		results = []

		for item in tqdm(result):
			if start_time==0:
				start_time = time.time()
			
			pooling_embedding=item["pooling_emb"] #a list of layer embedding with size (layer_num, seq_length, hidden_size)
			emb=item["emb"]
			docid=item["docid"]



			pooling_embedding = pooling_embedding.astype('float16')
			pooling_embedding = pooling_embedding.reshape(-1).tostring()
			pooling_embedding_tf = tf.train.Feature(bytes_list=tf.train.BytesList(value=[pooling_embedding]))
			if DOC_TYPE ==0:
				emb = emb.astype('float16')
				term_weights = LA.norm(emb, axis=-1)
				term_weights = term_weights.reshape(-1).tostring()
				term_weights_tf = tf.train.Feature(bytes_list=tf.train.BytesList(value=[term_weights]))



			docid_tf = tf.train.Feature(
							int64_list=tf.train.Int64List(value=[docid])
						)
			if DOC_TYPE ==0:
				features = tf.train.Features(feature={
								'doc_emb': pooling_embedding_tf,
								'term_weight':term_weights_tf,
								'docid':docid_tf
								}
							)
			else:
				features = tf.train.Features(feature={
								'doc_emb': pooling_embedding_tf,
								'docid':docid_tf
								}
							)
			example = tf.train.Example(features=features)
			writer.write(example.SerializeToString())

			counter+=1
			if (counter%100000==0):
				tf.logging.warn("Read {} examples in {} secs.".format(
					counter, int(time.time() - start_time)))

			# if counter==Save_SAMPLE_NUM :
			# 	writer.close()
			# 	counter=0
			# 	counter1+=1
			# 	writer = tf.python_io.TFRecordWriter(
			# 		OUTPUT_DIR + "/" + EMBEDDING_FILE + str(counter1) +".tf")
			# 	tf.logging.warn("Output next {} examples".format(
			# 		Save_SAMPLE_NUM))
		end_time = time.time()
		tf.logging.warn(" Latency = %.4f", (end_time-start_time)/counter)

		writer.close()
		

if __name__ == "__main__":
	tf.app.run()