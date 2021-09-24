import argparse
from util import read_rank_list, read_corpus, write_triplet_to_tf_record, output_conversation
import random
import tensorflow.compat.v1 as tf
import tokenization
import os
from progressbar import *
from pyserini.search import SimpleSearcher
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--qa_folder", required=True)
	parser.add_argument("--cqr_folder", required=True)
	parser.add_argument("--golden_ir_file", type=str, required=True, help='golden doc rank list by reranker (ColBERT)')
	parser.add_argument("--corpus_file", type=str, required=True, help='corpus file with format id\tcontent or pyserini index file')
	parser.add_argument("--vocab_file", type=str, required=True, help='BERT vocab file')
	parser.add_argument("--output_folder", type=str, required=True, help='tf record output folder')
	parser.add_argument("--max_context_length", type=int, default=100, help='context token length: without response we use 100')
	parser.add_argument("--repetition", type=int, default=100, help='tf record output folder')
	parser.add_argument("--add_response", action='store_true')
	parser.add_argument("--topk_rel", type=int, default=3, help='topk passages as rel from golden ir file')
	parser.add_argument("--topk_neg", type=int, default=200, help='topk passages as hard negatives from golden ir file')
	args = parser.parse_args()
	print('Preprocess...')
	conversation = output_conversation(args.qa_folder, args.cqr_folder)
	print('Loading Tokenizer...')
	tokenizer = tokenization.FullTokenizer(
		vocab_file=args.vocab_file, do_lower_case=True)
	if not os.path.exists(args.output_folder):
		os.mkdir(args.output_folder)
	writer = tf.python_io.TFRecordWriter(
		os.path.join(args.output_folder, 'dataset_train_tower.tf'))
	# conversation = read_query(args.cqa_file)
	qid_to_golden_docids = read_rank_list(args.golden_ir_file)
	# docid_to_doc = read_corpus(args.corpus_file)
	print('Read CAsT2019 index for text content')
	index = SimpleSearcher.from_prebuilt_index('cast2019')

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
					if args.add_response:
						context+=[turn['canonical_result']]
				else:
					context+=[raw_query]
					raw_query = '|'.join(context)
					if args.add_response:
						context+=[turn['canonical_result']]

				rewrite_query = turn['manual_rewritten_utterance']
				try:
					pos_docid = random.sample(qid_to_golden_docids[qid][:args.topk_rel], 1)[0] #pseudo label
					neg_docid = random.sample(qid_to_golden_docids[qid][:args.topk_neg], 1)[0] #hard negatives
				except:
					# print('no negatives')
					continue

				pos_doc = index.doc(pos_docid).raw() #docid_to_doc[pos_docid]
				neg_doc = index.doc(neg_docid).raw()  #docid_to_doc[neg_docid]
				write_triplet_to_tf_record(writer,
								   tokenizer=tokenizer,
								   raw_query=raw_query,
								   rewrite_query=rewrite_query,
								   docs=[neg_doc, pos_doc],
								   labels=[0 ,1],
								   max_context_length=args.max_context_length)
		pbar.update(10 * i + 1)

if __name__ == '__main__':
	main()