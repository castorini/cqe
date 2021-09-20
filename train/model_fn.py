from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from loss import batch_max_sim_softmax_loss, compute_max_sim, compute_max_sim_score
import modeling
import optimization

def term_reweight(embedding, mask, term_weights, dim, batch_size, seq_length, pooling, normalize):
		mask = tf.dtypes.cast(mask, tf.float32)
		mask = mask*term_weights
		mask = tf.tile(mask, (dim,1)) #[dim, batch_size, seq_length]
		mask = tf.reshape(mask, (-1, batch_size, seq_length)) #[dim, batch_size, seq_length]
		mask = tf.transpose(mask, perm=[1, 2, 0]) #[batch_size, seq_length, dim]

		embedding = embedding*mask
		if pooling =='max_min':
			embedding = max_min_pooling(embedding)
		if pooling =='average':
			embedding = tf.reduce_sum(embedding, axis=-2, keepdims=True) #[batch_size, 1, dim]
			mask = tf.reduce_sum(mask, axis=-2, keepdims=True) + 1e-15 #[batch_size, 1, dim]
			embedding = embedding/mask
		if normalize:
			embedding = tf.nn.l2_normalize(embedding, dim = -1)
		return embedding

def create_bert(bert_config, is_training, input_ids, input_mask, effective_mask, segment_ids,
				use_one_hot_embeddings, max_seq_len, output_dim, doc_type, pooling ,normalize):

	def filter(embedding, mask, effective_num, dim, batch_size, seq_length, pooling, normalize):
		mask = tf.where(tf.equal(mask, effective_num), tf.ones_like(mask), tf.zeros_like(mask))
		mask = tf.tile(mask, (dim,1)) #[dim, batch_size, seq_length]
		mask = tf.reshape(mask, (-1, batch_size, seq_length)) #[dim, batch_size, seq_length]
		mask = tf.transpose(mask, perm=[1, 2, 0]) #[batch_size, seq_length, dim]
		mask = tf.dtypes.cast(mask, tf.float32)
		embedding = embedding*mask
		if pooling =='max_min':
			embedding = max_min_pooling(embedding)
		if pooling =='average':
			embedding = tf.reduce_sum(embedding, axis=-2, keepdims=True) #[batch_size, 1, dim]
			mask = tf.reduce_sum(mask, axis=-2, keepdims=True) + 1e-15 #[batch_size, 1, dim]
			embedding = embedding/mask
		if normalize:
			embedding = tf.nn.l2_normalize(embedding, dim = -1)
		return embedding

	"""Creates a classification model."""
	model = modeling.BertModel(
		config=bert_config,
		is_training=is_training,
		input_ids=input_ids,
		input_mask=input_mask,
		token_type_ids=segment_ids,
		use_one_hot_embeddings=use_one_hot_embeddings)

	output_embedding = model.get_all_encoder_layers()[-1] #[batch_size, seq_length, hidden_size]
	cls_embedding = output_embedding[:,0:1,:] #[batch_size, 1, hidden_size]
	hidden_size = output_embedding.shape[-1].value
	seq_length = output_embedding.shape[-2].value
	batch_size = output_embedding.shape[0].value



	if output_dim!=768:
		output_weights = tf.get_variable(
			"output_weights"+str(output_dim), [output_dim, hidden_size],
			initializer=tf.truncated_normal_initializer(stddev=0.02))

		output_bias = tf.get_variable(
			"output_bias"+str(output_dim), [output_dim], initializer=tf.zeros_initializer())
		contextual_embedding = tf.matmul(output_embedding, output_weights, transpose_b=True)
		contextual_embedding = tf.nn.bias_add(contextual_embedding, output_bias)
	else:
		contextual_embedding = output_embedding

	# Filter out the first four tokens [CLS] [ Q(D) ]
	if doc_type=="Query":
		# contextual_embedding = filter(contextual_embedding[:,4:,:],
		# 						effective_mask[:,4:], 1,
		# 						output_dim, batch_size, seq_length-4, pooling, normalize)
		contextual_embedding = filter(contextual_embedding,
								effective_mask, 1,
								output_dim, batch_size, seq_length, pooling, normalize)

	if doc_type=='Doc':
		contextual_embedding = filter(contextual_embedding[:,4:,:],
									effective_mask[:,4:], 1,
									output_dim, batch_size, seq_length-4, pooling, normalize)


	input_length=tf.reduce_sum(effective_mask, axis=1)
	return output_embedding, contextual_embedding, cls_embedding, input_length



def create_model(bert_config, is_training, is_eval, is_output, input_ids, input_mask, segment_ids, effective_mask, label,
				use_one_hot_embeddings,
				colbert_dim, dotbert_dim, max_q_len, max_p_len, doc_type,
				loss, kd_source, train_model, eval_model):
	"""Creates a classification model."""

	with tf.variable_scope("Teacher") as scope:

		if (is_training) or (is_eval):
			_, query_emb, _, query_length=create_bert(bert_config, is_training, input_ids[0], input_mask[0], effective_mask[0], segment_ids[0],
											use_one_hot_embeddings, max_q_len, colbert_dim, 'Query', pooling=False, normalize=False)
			scope.reuse_variables()
			_, doc0_emb, doc0_cls_emb, doc0_length=create_bert(bert_config, is_training, input_ids[1], input_mask[1], effective_mask[1], segment_ids[1],
											use_one_hot_embeddings, max_p_len, colbert_dim, 'Doc', pooling=False, normalize=True)

		if is_training:
			scope.reuse_variables()
			_, doc1_emb, doc1_cls_emb, doc1_length=create_bert(bert_config, is_training, input_ids[2], input_mask[2], effective_mask[2], segment_ids[2],
											use_one_hot_embeddings, max_p_len, colbert_dim, 'Doc', pooling=False, normalize=True)


	with tf.variable_scope("query_reformulator") as scope:


		raw_query_emb, raw_query_pooling_emb, raw_query_cls_emb, raw_query_length=create_bert(bert_config, is_training, input_ids[3], input_mask[3], effective_mask[3], segment_ids[3],
										use_one_hot_embeddings, max_q_len, dotbert_dim, 'Query', pooling='average', normalize=False)
		# Term weight
		# term_weights = tf.get_variable(
		# 	"term_weights", [1, dotbert_dim],
		# 	initializer=tf.truncated_normal_initializer(stddev=0.02))

		# term_bias = tf.get_variable(
		# 	"term_bias", [1], initializer=tf.zeros_initializer())
		# term_weights = tf.matmul(term_weight_embedding, term_weights, transpose_b=True)
		# term_weights = tf.nn.bias_add(term_weights, term_bias) #[batch_size, seq_length, 1]
		# term_weights = tf.nn.sigmoid(term_weights)
		# term_weights = tf.reduce_mean(term_weights, axis=-1) #[batch_size, seq_length]

		if is_output and (doc_type==0):
			# if doc_type==0: #query
			# term_weight_embedding, contextual_emb, _, doc_length=create_bert(bert_config, is_training, input_ids[3], input_mask[3], effective_mask[3], segment_ids[3],
			# 					use_one_hot_embeddings, max_q_len, dotbert_dim, 'Query', pooling='average', normalize=False)
			return raw_query_pooling_emb ,raw_query_emb, raw_query_length
	with tf.variable_scope("Student") as scope:

		query_embedding, query_pooling_emb, _, query_length=create_bert(bert_config, is_training, input_ids[3], input_mask[3], effective_mask[3], segment_ids[3],
											use_one_hot_embeddings, max_q_len, dotbert_dim, 'Query', pooling='average', normalize=False)
		scope.reuse_variables()
		doc0_embedding, doc0_pooling_emb, doc0_cls_emb, doc0_length=create_bert(bert_config, is_training, input_ids[1], input_mask[1], effective_mask[1], segment_ids[1],
						use_one_hot_embeddings, max_p_len, dotbert_dim, 'Doc', pooling='average', normalize=False)

		if is_training:
			scope.reuse_variables()
			doc1_embedding, doc1_pooling_emb, doc1_cls_emb, doc1_length=create_bert(bert_config, is_training, input_ids[2], input_mask[2], effective_mask[2], segment_ids[2],
							use_one_hot_embeddings, max_p_len, dotbert_dim, 'Doc', pooling='average', normalize=False)
		if is_output and (doc_type==1): #document
			# term_weight_embedding, contextual_emb, _, doc_length=create_bert(bert_config, is_training, input_ids[1], input_mask[1], effective_mask[1], segment_ids[1],
			# 					use_one_hot_embeddings, max_p_len, dotbert_dim, 'Doc', pooling='average', normalize=False)

			return doc0_pooling_emb ,doc0_embedding, doc0_length



	with tf.variable_scope("loss"):
		score = 0
		batch_size = raw_query_emb.shape[0].value
		seq_length = raw_query_emb.shape[1].value
		dim = raw_query_emb.shape[2].value
		# query_pooling_emb = term_reweight(output_embedding, effective_mask[3], term_weights, dim, batch_size, seq_length, 'average', False)
		if is_training:


			temperature = 0.25
			kl = tf.keras.losses.KLDivergence()
			tct_teacher_loss, tct_teacher_logits, tct_teacher_margins = batch_max_sim_softmax_loss(query_emb, doc0_emb, doc1_emb, 4*batch_size)
			tct_teacher_logits = tf.nn.softmax(tct_teacher_logits/temperature, axis=-1)


			tct_student_loss, tct_student_logits, tct_student_margins = batch_max_sim_softmax_loss(raw_query_pooling_emb, doc0_pooling_emb, doc1_pooling_emb, 4*batch_size) #
			tct_student_logits = tf.nn.softmax(tct_student_logits, axis=-1)

			tct_teacher_student_kl_loss = kl(tct_teacher_logits, tct_student_logits)
			tct_teacher_student_mse_loss = tf.keras.losses.MSE(tct_teacher_margins, tct_student_margins)


			loss = tf.reduce_mean(tct_teacher_student_kl_loss)

		else:
			if eval_model=='student':
				# Student
				score = compute_max_sim_score([raw_query_pooling_emb], [doc0_pooling_emb])
			elif eval_model=='teacher':
				# Teacher
				score = compute_max_sim_score([query_emb], [doc0_emb])
		return loss, score, query_length


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
					 num_train_steps, num_warmup_steps, use_tpu,
					 use_one_hot_embeddings,
					 colbert_dim, dotbert_dim, max_q_len, max_p_len, doc_type,
					 loss, kd_source, train_model, eval_model,
					 is_eval, is_output):
	"""Returns `model_fn` closure for TPUEstimator."""
	def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
		"""The `model_fn` for TPUEstimator."""
		tf.logging.info("*** Features ***")
		for name in sorted(features.keys()):
			tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

		is_training = (mode == tf.estimator.ModeKeys.TRAIN)

		input_ids=[]
		input_mask=[]
		segment_ids=[]
		mask_lm_info = []
		if is_training:
			input_ids = [features["rewrite_query_ids"], features["doc0_ids"], features["doc1_ids"], features["raw_query_ids"]]
			input_mask = [features["rewrite_query_mask"], features["doc0_mask"], features["doc1_mask"], features["raw_query_mask"]]
			segment_ids = [features["rewrite_query_segment_ids"], features["doc0_segment_ids"], features["doc1_segment_ids"], features["raw_query_segment_ids"]]
			effective_mask = [features["effective_rewrite_query_mask"], features["effective_doc0_mask"], features["effective_doc1_mask"], features["effective_raw_query_mask"]]
		elif is_eval:
			input_ids = [features["query_ids"], features["docx_ids"], 0, features["query_ids"]]
			input_mask = [features["query_mask"], features["docx_mask"], 0, features["query_mask"]]
			segment_ids = [features["query_segment_ids"], features["docx_segment_ids"], 0, features["query_segment_ids"]]
			effective_mask = [features["effective_query_mask"], features["effective_docx_mask"], 0, features["effective_query_mask"]]
		elif is_output:
			input_ids=[features["input_ids"], features["input_ids"], features["input_ids"], features["input_ids"]]
			input_mask = [features["input_mask"], features["input_mask"], features["input_mask"], features["input_mask"]]
			segment_ids = [features["segment_ids"], features["segment_ids"], features["segment_ids"], features["segment_ids"]]
			effective_mask = [features["effective_input_mask"], features["effective_input_mask"], features["effective_input_mask"], features["effective_input_mask"]]



		label = features["label"]


		tf.logging.info("Create model")
		if (is_training) or (is_eval):
			(total_loss, score, doc_length) = create_model(
				bert_config, is_training, is_eval, is_output, input_ids, input_mask, segment_ids, effective_mask, label, use_one_hot_embeddings,
				colbert_dim, dotbert_dim, max_q_len, max_p_len, doc_type, loss, kd_source, train_model, eval_model)
		elif is_output:
			(pooling_emb, emb, doc_length) = create_model(
				bert_config, is_training, is_eval, is_output, input_ids, input_mask, segment_ids, effective_mask, label, use_one_hot_embeddings,
				colbert_dim, dotbert_dim, max_q_len, max_p_len, doc_type, loss, kd_source, train_model, eval_model)

		tf.logging.info("Finish create model")
		tvars = tf.trainable_variables()

		scaffold_fn = None
		if init_checkpoint:
			(assignment_map, initialized_variable_names)= modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
			(assignment_map1, initialized_variable_names1) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint, 'Student/', 'query_reformulator/')
			assignment_maps = [assignment_map, assignment_map1]
			initialized_variable_names.update(initialized_variable_names1)

			tf.logging.info("**** Assignment Map ****")
			if use_tpu:
				def tpu_scaffold():
					for assignment_map in assignment_maps:
					  tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
					return tf.train.Scaffold()

				scaffold_fn = tpu_scaffold
			else:
				tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
		tf.logging.info("**** Trainable Variables ****")

		for var in tvars:
			init_string = ""
			if var.name in initialized_variable_names:
				init_string = ", *INIT_FROM_CKPT*"
			tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
							init_string)

		output_spec = None
		if mode == tf.estimator.ModeKeys.TRAIN:
			train_op = optimization.create_optimizer(
						total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu, train_model)

			output_spec = tf.contrib.tpu.TPUEstimatorSpec(
						mode=mode,
						loss=total_loss,
						train_op=train_op,
						scaffold_fn=scaffold_fn)

		elif mode == tf.estimator.ModeKeys.PREDICT:
			if is_output:
				output_spec = tf.contrib.tpu.TPUEstimatorSpec(
								mode=mode,
								predictions={
									"docid": features['docid'],
									"pooling_emb":pooling_emb,
									"emb":emb,
									"doc_length":doc_length,
								},
								scaffold_fn=scaffold_fn)
			elif is_eval:
				output_spec = tf.contrib.tpu.TPUEstimatorSpec(
								mode=mode,
								predictions={
									"log_probs": score,
									"label_ids": label,
								},
								scaffold_fn=scaffold_fn)

		else:
			raise ValueError(
					"Only TRAIN and PREDICT modes are supported: %s" % (mode))

		return output_spec

	return model_fn

def input_fn_builder(dataset_path, max_q_len, max_p_len, doc_type,
					 is_training, is_eval, is_output, max_eval_examples=None):
	"""Creates an `input_fn` closure to be passed to TPUEstimator."""
	def input_fn(params):
		"""The actual input function."""
		batch_size = params["batch_size"]
		output_buffer_size = batch_size * 1000
		max_seq_len = {0: max_q_len, 1:max_p_len}


		def extract_fn(data_record):
			if is_training:
				features = {
							"raw_query_ids": tf.FixedLenSequenceFeature(
								[], tf.int64, allow_missing=True),
							"raw_query_mask": tf.FixedLenSequenceFeature(
								[], tf.int64, allow_missing=True),
							"raw_query_segment_ids": tf.FixedLenSequenceFeature(
								[], tf.int64, allow_missing=True),
							"rewrite_query_ids": tf.FixedLenSequenceFeature(
								[], tf.int64, allow_missing=True),
							"rewrite_query_mask": tf.FixedLenSequenceFeature(
								[], tf.int64, allow_missing=True),
							"rewrite_query_segment_ids": tf.FixedLenSequenceFeature(
								[], tf.int64, allow_missing=True),
							"doc_ids0": tf.FixedLenSequenceFeature(
								[], tf.int64, allow_missing=True),
							"doc_ids1": tf.FixedLenSequenceFeature(
								[], tf.int64, allow_missing=True),
							"label": tf.FixedLenFeature([], tf.int64),
							}
				sample = tf.parse_single_example(data_record, features)

				raw_query_ids = tf.cast(sample["raw_query_ids"][:max_q_len], tf.int32)
				raw_query_segment_ids  = tf.cast(sample["raw_query_segment_ids"][:max_q_len], tf.int32)
				raw_query_mask = tf.ones_like(raw_query_ids)
				# effective_raw_query_mask = tf.ones_like(query_ids)
				effective_raw_query_mask = tf.cast(sample["raw_query_mask"][:max_q_len], tf.int32)

				rewrite_query_ids = tf.cast(sample["rewrite_query_ids"][:max_q_len], tf.int32)
				rewrite_query_segment_ids  = tf.cast(sample["rewrite_query_segment_ids"][:max_q_len], tf.int32)
				rewrite_query_mask = tf.ones_like(rewrite_query_ids)
				effective_rewrite_query_mask = tf.cast(sample["rewrite_query_mask"][:max_q_len], tf.int32)


				doc0_ids = tf.cast(sample["doc_ids0"][:max_p_len], tf.int32)
				doc0_segment_ids = tf.zeros_like(doc0_ids)
				doc0_mask = tf.ones_like(doc0_ids)
				effective_doc0_mask = tf.ones_like(doc0_ids)


				doc1_ids = tf.cast(sample["doc_ids1"][:max_p_len], tf.int32)
				doc1_segment_ids = tf.zeros_like(doc1_ids)
				doc1_mask = tf.ones_like(doc1_ids)
				effective_doc1_mask =  tf.ones_like(doc1_ids)


				label = tf.cast(sample["label"], tf.float32)


				features = {
					"raw_query_ids": raw_query_ids,
					"raw_query_segment_ids": raw_query_segment_ids,
					"raw_query_mask": raw_query_mask,
					"effective_raw_query_mask": effective_raw_query_mask,
					"rewrite_query_ids": rewrite_query_ids,
					"rewrite_query_segment_ids": rewrite_query_segment_ids,
					"rewrite_query_mask": rewrite_query_mask,
					"effective_rewrite_query_mask": effective_rewrite_query_mask,
					"doc0_ids": doc0_ids,
					"doc0_segment_ids": doc0_segment_ids,
					"doc0_mask": doc0_mask,
					"effective_doc0_mask": effective_doc0_mask,
					"doc1_ids": doc1_ids,
					"doc1_segment_ids": doc1_segment_ids,
					"doc1_mask": doc1_mask,
					"effective_doc1_mask": effective_doc1_mask,
					"label": label,
					}
			else:
				if is_output:
					if doc_type==0:
						features = {
							"raw_query_ids": tf.FixedLenSequenceFeature(
									[], tf.int64, allow_missing=True),
							"raw_query_mask": tf.FixedLenSequenceFeature(
									[], tf.int64, allow_missing=True),
							"raw_query_segment_ids": tf.FixedLenSequenceFeature(
									[], tf.int64, allow_missing=True),
							"query_id": tf.FixedLenFeature([], tf.int64),
						}
						sample = tf.parse_single_example(data_record, features)
						input_ids = tf.cast(sample["raw_query_ids"][:max_seq_len[doc_type]], tf.int32)
						segment_ids = tf.cast(sample["raw_query_segment_ids"][:max_seq_len[doc_type]], tf.int32)
						input_mask = tf.ones_like(input_ids)
						effective_input_mask = tf.cast(sample["raw_query_mask"][:max_seq_len[doc_type]], tf.int32)
						docid = tf.cast(sample["query_id"], tf.int32)
						label = tf.cast(0, tf.int32) #dummy
						features = {
							"input_ids": input_ids,
							"segment_ids": segment_ids,
							"input_mask": input_mask,
							"effective_input_mask": effective_input_mask,
							"docid": docid,
							"label": label,
						}
					elif doc_type==1:
						features = {
							"doc_ids": tf.FixedLenSequenceFeature(
								[], tf.int64, allow_missing=True),
							"docid": tf.FixedLenFeature([], tf.int64),
						}
						sample = tf.parse_single_example(data_record, features)
						doc_ids = sample["doc_ids"][:max_seq_len[doc_type]]

						input_ids = tf.cast(doc_ids, tf.int32)
						segment_ids = tf.zeros_like(input_ids)
						input_mask = tf.ones_like(input_ids)
						docid = tf.cast(sample["docid"], tf.int32)
						label = tf.cast(0, tf.int32) #dummy
						features = {
							"input_ids": input_ids,
							"segment_ids": segment_ids,
							"input_mask": input_mask,
							"effective_input_mask": input_mask,
							"docid": docid,
							"label": label,
						}
				elif is_eval:
					features = {
						"query_ids": tf.FixedLenSequenceFeature(
							[], tf.int64, allow_missing=True),
						"query_segment_ids": tf.FixedLenSequenceFeature(
							[], tf.int64, allow_missing=True),
						"query_mask": tf.FixedLenSequenceFeature(
							[], tf.int64, allow_missing=True),
						"doc_ids": tf.FixedLenSequenceFeature(
							[], tf.int64, allow_missing=True),
						"label": tf.FixedLenFeature([], tf.int64),
					}
					sample = tf.parse_single_example(data_record, features)

					query_ids = tf.cast(sample["query_ids"][:max_q_len], tf.int32)
					query_segment_ids = tf.cast(sample["query_segment_ids"][:max_q_len], tf.int32)
					# query_segment_ids = tf.zeros_like(query_ids)
					query_mask = tf.ones_like(query_ids)
					effective_query_mask = tf.cast(sample["query_mask"][:max_q_len], tf.int32)
					# effective_query_mask = tf.ones_like(query_ids)

					docx_ids = tf.cast(sample["doc_ids"][:max_p_len], tf.int32)
					docx_segment_ids = tf.zeros_like(docx_ids)
					docx_mask = tf.ones_like(docx_ids)
					effective_docx_mask = tf.ones_like(docx_ids)


					label = tf.cast(sample["label"], tf.int32)

					features = {
						"query_ids": query_ids,
						"query_segment_ids": query_segment_ids,
						"query_mask": query_mask,
						"effective_query_mask": effective_query_mask,
						"docx_ids": docx_ids,
						"docx_segment_ids": docx_segment_ids,
						"docx_mask": docx_mask,
						"effective_docx_mask": effective_docx_mask,
						"label": label,
					}


			return features

		dataset = tf.data.TFRecordDataset([dataset_path])
		dataset = dataset.map(
			extract_fn, num_parallel_calls=4).prefetch(output_buffer_size)

		if is_training:
			dataset = dataset.repeat()
			dataset = dataset.shuffle(buffer_size=1000)
			dataset = dataset.padded_batch(
						batch_size=batch_size,
						padded_shapes={
							"raw_query_ids": [max_q_len],
							"raw_query_segment_ids": [max_q_len],
							"raw_query_mask": [max_q_len],
							"effective_raw_query_mask": [max_q_len],
							"rewrite_query_ids": [max_q_len],
							"rewrite_query_segment_ids": [max_q_len],
							"rewrite_query_mask": [max_q_len],
							"effective_rewrite_query_mask": [max_q_len],
							"doc0_ids": [max_p_len],
							"doc0_segment_ids": [max_p_len],
							"doc0_mask": [max_p_len],
							"effective_doc0_mask": [max_p_len],
							"doc1_ids": [max_p_len],
							"doc1_segment_ids": [max_p_len],
							"doc1_mask": [max_p_len],
							"effective_doc1_mask": [max_p_len],
							"label": []
						},
						padding_values={
							"raw_query_ids": 0,
							"raw_query_segment_ids": 0,
							"raw_query_mask": 0,
							"effective_raw_query_mask":0,
							"rewrite_query_ids": 0,
							"rewrite_query_segment_ids": 0,
							"rewrite_query_mask": 0,
							"effective_rewrite_query_mask":0,
							"doc0_ids": 0,
							"doc0_segment_ids": 0,
							"doc0_mask": 0,
							"effective_doc0_mask": 0,
							"doc1_ids": 0,
							"doc1_segment_ids": 0,
							"doc1_mask": 0,
							"effective_doc1_mask": 0,
							"label": 0.0,
						},
						drop_remainder=True)
		else:
			if max_eval_examples:
				# Use at most this number of examples (debugging only).
				dataset = dataset.take(max_eval_examples)
			if is_output:
				dataset = dataset.padded_batch(
							batch_size=batch_size,
							padded_shapes={
								"input_ids": [max_seq_len[doc_type]],
								"segment_ids": [max_seq_len[doc_type]],
								"input_mask": [max_seq_len[doc_type]],
								"effective_input_mask": [max_seq_len[doc_type]],
								"docid": [],
								"label": [],
							},
							padding_values={
								"input_ids": 0,
								"segment_ids": 0,
								"input_mask": 0,
								"effective_input_mask": 0,
								"docid": 0,
								"label": 0,
							},
							drop_remainder=True)

			elif is_eval:
				dataset = dataset.padded_batch(
								batch_size=batch_size,
								padded_shapes={
									"query_ids": [max_q_len],
									"query_segment_ids": [max_q_len],
									"query_mask": [max_q_len],
									"effective_query_mask": [max_q_len],
									"docx_ids": [max_p_len],
									"docx_segment_ids": [max_p_len],
									"docx_mask": [max_p_len],
									"effective_docx_mask": [max_p_len],
									"label": []
								},
								padding_values={
									"query_ids": 0,
									"query_segment_ids": 0,
									"query_mask": 0,
									"effective_query_mask": 0,
									"docx_ids": 0,
									"docx_segment_ids": 0,
									"docx_mask": 0,
									"effective_docx_mask":0,
									"label": 0,
								},
								drop_remainder=True)
		return dataset
	return input_fn