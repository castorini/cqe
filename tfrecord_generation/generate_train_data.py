import argparse
from util import read_query, read_rank_list, read_corpus, write_triplet_to_tf_record
import random
import tensorflow.compat.v1 as tf
import tokenization
from progressbar import *
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--conversation_file", type=str, required=True, help='json file')
	parser.add_argument("--rewrite_query_file", type=str, required=True, help='id\tquery')
	parser.add_argument("--query_golden_ir_file", type=str, required=True, help='query doc rank list')
	parser.add_argument("--query_ir_file", type=str, required=True, help='query doc rank list')
	parser.add_argument("--corpus_file", type=str, required=True, help='corpus file')
	parser.add_argument("--vocab_file", type=str, required=True, help='BERT vocab file')
	parser.add_argument("--output_folder", type=str, required=True, help='tf record output folder')
	parser.add_argument("--repetition", type=int, default=100, help='tf record output folder')
	args = parser.parse_args()
	print('Loading Tokenizer...')
	tokenizer = tokenization.FullTokenizer(
		vocab_file=args.vocab_file, do_lower_case=True)
	writer = tf.python_io.TFRecordWriter(
		args.output_folder + '/dataset_train_tower.tf')
	conversation = read_query(args.conversation_file)
	qid_to_golden_docids = read_rank_list(args.query_golden_ir_file)
	qid_to_docids = read_rank_list(args.query_ir_file)
	docid_to_doc = read_corpus(args.corpus_file)
	qid_to_rewrite_query = read_corpus(args.rewrite_query_file)

	widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),
		' ', ETA(), ' ', FileTransferSpeed()]
	pbar = ProgressBar(widgets=widgets, maxval=10*args.repetition).start()
	for i in range(args.repetition):
		for session in conversation:
			session_id = str(session['number'])
			for _, turn in enumerate(session['turn']):
				turn_id = str(turn['number'])
				qid = session_id + '_' + turn_id
				raw_query = turn['raw_utterance'].strip()
				if turn_id == '1':
					context = [raw_query]
				else:
					context+=[raw_query]
					raw_query = '|'.join(context)

				rewrite_query = qid_to_rewrite_query[qid]

				try:
					pos_docid = random.sample(qid_to_golden_docids[qid][:3], 1)[0] #sudo label
					neg_docid = random.sample(qid_to_docids[qid][:200], 1)[0]
				except:
					# print('no negatives')
					continue

				pos_doc = docid_to_doc[pos_docid]
				neg_doc = docid_to_doc[neg_docid]
				write_triplet_to_tf_record(writer,
								   tokenizer=tokenizer,
								   raw_query=raw_query,
								   rewrite_query=rewrite_query,
								   docs=[neg_doc, pos_doc],
								   labels=[0 ,1])
		pbar.update(10 * i + 1)

if __name__ == '__main__':
	main()