import argparse
from utility import read_query, read_rank_list, read_corpus, write_to_tf_record
from random import choices
import tensorflow
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--conversation_file", type=str, required=True, help='json file')
	parser.add_argument("--rewrite_query_file", type=str, required=True, help='id\tquery')
	parser.add_argument("--query_ir_file", type=str, required=True, help='query doc rank list')
	parser.add_argument("--corpus_file", type=str, required=True, help='corpus file')
	parser.add_argument("--vocab_file", type=str, required=True, help='BERT vocab file')
	parser.add_argument("--output_folder", type=str, required=True, help='tf record output folder')

	writer = tf.python_io.TFRecordWriter(
		args.output_folder + '/dataset_train_tower.tf')
	conversation = read_query(args.conversation_file)
	qid_to_docids = read_rank_list(args.query_ir_file)
	docid_to_doc = read_corpus(args.corpus_file)
	qid_to_rewrite_query = read_corpus(args.rewrite_query_file)

	print('Loading Tokenizer...')
	tokenizer = tokenization.FullTokenizer(
		vocab_file=args.vocab_file, do_lower_case=True)


	for session in data:
		session_id = str(session['number'])
		for _, conversation in enumerate(session['turn']):
			turn_id = str(conversation['number'])
			qid = session_id + '_' + turn_id
			raw_query = conversation['raw_utterance'].strip()
			if turn_id == '1':
				context = []
			else:
				context+=[raw_query]
				raw_query = '|'.join(context)
				import pdb; pdb.set_trace()  # breakpoint c3857184 //

			rewrite_query = qid_to_rewrite_query[qid]
			docids = qid_to_docids[qid]

			pos_docid = random.sample(docids[:3], 1) #sudo label
			neg_docid = random.sample(docids[:200], 1)
			pos_doc = docid_to_doc[pos_docid]
			neg_doc = docid_to_doc[neg_docid]
			write_to_tf_record(writer,
							   tokenizer=tokenizer,
							   raw_query=raw_query,
							   rewrite_query=rewrite_query,
							   docs=[neg_doc, pos_doc],
							   labels=[0 ,1])

if __name__ == '__main__':
	main()