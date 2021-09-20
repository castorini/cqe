import json
import collections
import tokenization
import tensorflow as tf
def read_cast_query(file, use_response=False, context_type=None):
	print('Read query json file...')
	with open(file) as json_file:
		data = json.load(json_file)
	qid_to_query = {}
	for session in data:
		session_num = str(session['number'])
		hist_query = []
		hist_response = []
		hist_context = []

		for turn_id, conversations in enumerate(session['turn']):
			if use_response:
				pid = conversations['canonical_result_id'] + '-' + str(conversations['passage_id'])
			query = conversations['raw_utterance'].strip()

			conversation_num = str(conversations['number'])
			qid=session_num+"_"+conversation_num


			hist_context.append(query)
			if use_response:
				hist_context.append(response)
			hist_query.append(query)

			if len(hist_query)==1:
				input_query = query
			else:
				if context_type == 'all':
					input_query = '|'.join(hist_context)
				elif context_type == 'last-only':
					input_query = '|'.join(hist_query[:-1]+hist_context[-2:])
				else:
					input_query = '|'.join(hist_query)
			qid_to_query[qid] = input_query
	return qid_to_query

def read_rank_list(file):
	print('Read rank list...')
	qid_to_docid = collections.defaultdict(list)
	with open(file) as f:
		for line in f:
			line = line.strip().split('\t')
			qid = line[0]
			docid = line[1]
			qid_to_docid[qid].append(docid)
	return qid_to_docid

def read_corpus(file):
	docid_to_doc = {}
	with open(file) as f:
		for line in f:
			try:
				line = line.strip().split('\t')
				docid = line[0]
				doc = line[1]
				docid_to_doc[docid] = doc
			except:
				print('no content')

	return docid_to_doc

def write_triplet_to_tf_record(writer, tokenizer, raw_query, rewrite_query, docs, labels,
					max_query_length=36, max_doc_length=154, ids_file=None, query_id=None, doc_ids=None, is_train=True):
	feature = {}
	raw_query = tokenization.convert_to_unicode(raw_query)
	if '|' in raw_query:
		context ='|'.join(raw_query.split('|')[:-1])
		raw_query = raw_query.split('|')[-1]
		raw_query_token_ids, raw_query_segment_ids, raw_query_mask = tokenization.convert_to_coversation_query(
			context=context, query='[Q] '+raw_query, max_context_length=100, max_query_length=max_query_length, tokenizer=tokenizer,
			add_cls=True, padding_mask=True)


	else:
		raw_query_token_ids = tokenization.convert_to_colbert_input(
			text='[Q] '+raw_query, max_seq_length=max_query_length, tokenizer=tokenizer,
			add_cls=True, padding_mask=True)
		raw_query_mask = [0]*4 + [1]*(len(raw_query_token_ids)-4)
		raw_query_segment_ids = [0]*1 + [1]*(len(raw_query_token_ids)-1)

	raw_query_token_ids_tf = tf.train.Feature(
		int64_list=tf.train.Int64List(value=raw_query_token_ids))
	raw_query_segment_ids_tf = tf.train.Feature(
		int64_list=tf.train.Int64List(value=raw_query_segment_ids))
	raw_query_mask_tf = tf.train.Feature(
		int64_list=tf.train.Int64List(value=raw_query_mask))


	feature['raw_query_ids']=raw_query_token_ids_tf
	feature['raw_query_segment_ids']=raw_query_segment_ids_tf
	feature['raw_query_mask']=raw_query_mask_tf

	rewrite_query_token_ids = tokenization.convert_to_colbert_input(
			text='[Q] '+rewrite_query, max_seq_length=max_query_length, tokenizer=tokenizer,
			add_cls=True, padding_mask=True)
	rewrite_query_mask = [0]*4 + [1]*(len(rewrite_query_token_ids)-4)
	rewrite_query_segment_ids = [0]*1 + [1]*(len(rewrite_query_token_ids)-1)

	rewrite_query_token_ids_tf = tf.train.Feature(
		int64_list=tf.train.Int64List(value=rewrite_query_token_ids))
	rewrite_query_segment_ids_tf = tf.train.Feature(
		int64_list=tf.train.Int64List(value=rewrite_query_segment_ids))
	rewrite_query_mask_tf = tf.train.Feature(
		int64_list=tf.train.Int64List(value=rewrite_query_mask))
	feature['rewrite_query_ids']=rewrite_query_token_ids_tf
	feature['rewrite_query_segment_ids']=rewrite_query_segment_ids_tf
	feature['rewrite_query_mask']=rewrite_query_mask_tf

	for i, (doc_text, label) in enumerate(zip(docs, labels)):
		doc_token_ids = tokenization.convert_to_colbert_input(
			text='[D] '+doc_text,
			max_seq_length=max_doc_length,
			tokenizer=tokenizer,
			add_cls=True, padding_mask=False)

		doc_ids_tf = tf.train.Feature(
			int64_list=tf.train.Int64List(value=doc_token_ids))

		labels_tf = tf.train.Feature(
			int64_list=tf.train.Int64List(value=[label]))

		if is_train:
			feature['doc_ids'+str(label)]=doc_ids_tf
		else:
			feature['doc_ids']=doc_ids_tf


		feature['label']=labels_tf
		if ids_file:
			ids_file.write('\t'.join([query_id, doc_ids[i]]) + '\n')
		if not is_train:
			features = tf.train.Features(feature=feature)
			example = tf.train.Example(features=features)
			writer.write(example.SerializeToString())
	if is_train:
		features = tf.train.Features(feature=feature)
		example = tf.train.Example(features=features)
		writer.write(example.SerializeToString())

def write_query_to_tf_record(writer, tokenizer, raw_query, qid, max_query_length=36):
	feature = {}
	raw_query = tokenization.convert_to_unicode(raw_query)
	if '|' in raw_query:
		context ='|'.join(raw_query.split('|')[:-1])
		raw_query = raw_query.split('|')[-1]
		raw_query_token_ids, raw_query_segment_ids, raw_query_mask = tokenization.convert_to_coversation_query(
			context=context, query='[Q] '+raw_query, max_context_length=100, max_query_length=max_query_length, tokenizer=tokenizer,
			add_cls=True, padding_mask=True)


	else:
		raw_query_token_ids = tokenization.convert_to_colbert_input(
			text='[Q] '+raw_query, max_seq_length=max_query_length, tokenizer=tokenizer,
			add_cls=True, padding_mask=True)
		raw_query_mask = [0]*4 + [1]*(len(raw_query_token_ids)-4)
		raw_query_segment_ids = [0]*1 + [1]*(len(raw_query_token_ids)-1)

	raw_query_token_ids_tf = tf.train.Feature(
		int64_list=tf.train.Int64List(value=raw_query_token_ids))
	raw_query_segment_ids_tf = tf.train.Feature(
		int64_list=tf.train.Int64List(value=raw_query_segment_ids))
	raw_query_mask_tf = tf.train.Feature(
		int64_list=tf.train.Int64List(value=raw_query_mask))
	qid_tf = tf.train.Feature(
      int64_list=tf.train.Int64List(value=[qid]))


	feature['raw_query_ids']=raw_query_token_ids_tf
	feature['raw_query_segment_ids']=raw_query_segment_ids_tf
	feature['raw_query_mask']=raw_query_mask_tf
	feature['query_id']=qid_tf


	features = tf.train.Features(feature=feature)
	example = tf.train.Example(features=features)
	writer.write(example.SerializeToString())

