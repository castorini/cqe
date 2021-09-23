import argparse
from util import read_cast_query, write_query_to_tf_record, read_corpus
import random
import tensorflow.compat.v1 as tf
import tokenization
import os
from progressbar import *
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--query_file", type=str, required=True, help='query file')
	parser.add_argument("--vocab_file", type=str, required=True, help='BERT vocab file')
	parser.add_argument("--output_folder", type=str, required=True, help='tf record output folder')
	parser.add_argument("--output_filename", type=str, default=None)
	parser.add_argument("--use_response", action='store_true')
	parser.add_argument("--max_context_length", type=int, default=100)
	args = parser.parse_args()
	if not os.path.exists(args.output_folder):
		os.mkdir(args.output_folder)
	print('Loading Tokenizer...')
	tokenizer = tokenization.FullTokenizer(
		vocab_file=args.vocab_file, do_lower_case=True)
	if args.output_filename == None:
		output_filename = '.'.join(args.query_file.split('.')[:-1])
		output_filename = os.path.join(args.output_folder, output_filename)
	else:
		output_filename = os.path.join(args.output_folder, args.output_filename)
	if args.use_response:
		from pyserini.search import SimpleSearcher
		print('Load index so that we can get system response ...')
		# Now we use cast2019 (and 2020) index as our default 
		index = SimpleSearcher.from_prebuilt_index('cast2019')
	else:
		index = None

	qid_to_query = read_cast_query(args.query_file, args.use_response, index)
	# qid_to_query = read_corpus(file)
	writer = tf.python_io.TFRecordWriter(output_filename+'.tf')
	text_writer = open(output_filename+'.tsv', 'w')
	id_writer = open(output_filename+'.id', 'w')
	widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),
		' ', ETA(), ' ', FileTransferSpeed()]
	pbar = ProgressBar(widgets=widgets, maxval=10*len(qid_to_query)).start()
	for i, qid in enumerate(qid_to_query.keys()):
		raw_query = qid_to_query[qid]
		write_query_to_tf_record(writer, tokenizer, raw_query, i, args.max_context_length)
		id_writer.write('%d\t%s\n'%(i,qid))
		text_writer.write('{}\t{}\n'.format(qid, raw_query))
		pbar.update(10 * i + 1)
	writer.close()
	text_writer.close()
	id_writer.close()

if __name__ == '__main__':
	main()