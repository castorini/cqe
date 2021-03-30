import argparse
from utility import read_corpus, write_query_to_tf_record
import random
import tensorflow.compat.v1 as tf
import tokenization
from progressbar import *
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--query_file", type=str, required=True, help='query file')
	parser.add_argument("--vocab_file", type=str, required=True, help='BERT vocab file')
	parser.add_argument("--output_file", type=str, required=True, help='tf record output folder')
	args = parser.parse_args()
	print('Loading Tokenizer...')
	tokenizer = tokenization.FullTokenizer(
		vocab_file=args.vocab_file, do_lower_case=True)
	
	qid_to_query = read_corpus(args.query_file)
	writer = tf.python_io.TFRecordWriter(
		args.output_file+'.tf')
	id_writer = open(args.output_file+'.id', 'w')
	widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),
		' ', ETA(), ' ', FileTransferSpeed()]
	pbar = ProgressBar(widgets=widgets, maxval=10*len(qid_to_query)).start()
	for i, qid in enumerate(qid_to_query.keys()):
		raw_query = qid_to_query[qid]

		write_query_to_tf_record(writer, tokenizer, raw_query.replace(' | ', '|'), i)
		id_writer.write('%d\t%s\n'%(i,qid))
		pbar.update(10 * i + 1)
	writer.close()
	id_writer.close()

if __name__ == '__main__':
	main()