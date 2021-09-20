"""
This code converts MS MARCO train, dev and eval tsv data into the tfrecord files
that will be consumed by BERT.
"""
import collections
import os
import glob
import re
import tensorflow.compat.v1 as tf
import time
# local module
import tokenization
import math
from tqdm import tqdm
from read_data import msmarco_corpus
from multiprocessing import Pool
import multiprocessing



flags = tf.flags

FLAGS = flags.FLAGS


flags.DEFINE_string(
    "output_folder", None,
    "Folder where the tfrecord files will be written.")

flags.DEFINE_string(
    "vocab_file",
    "./data/bert/uncased_L-24_H-1024_A-16/vocab.txt",
    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_filename",
    None,
    "output filename")

flags.DEFINE_string(
    "corpus_path",
    "./data/top1000.dev.tsv",
    "Path to the MSMARCO training dataset containing the tab separated "
    "<query, positive_paragraph, negative_paragraph> tuples.")

flags.DEFINE_string(
    "corpus_prefix",
    None,
    "corpus name prefix")
flags.DEFINE_string(
    "meta",
    "content",
    "meta information used to generate corpus")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string(
    "doc_type", "doc",
    "Denote the type of document, either doc or query")

flags.DEFINE_string(
    "data_format", "tsv",
    "corpus data format: tsv or json")

flags.DEFINE_integer(
    "batch_size", 100,
    "batch size for inference. We need to store the remaining to the last file ")

flags.DEFINE_integer(
    "num_eval_docs", 1000,
    "The maximum number of docs per query for dev and eval sets.")

flags.DEFINE_integer(
    "workers", None,
    "if None, workers are equal to number of files.")

if os.path.isdir(FLAGS.corpus_path):
    assert FLAGS.corpus_prefix != None, "You have to input corpus_prefix arg"
    if FLAGS.output_filename==None:
      FLAGS.output_filename = FLAGS.corpus_prefix
else:
  if FLAGS.output_filename==None:
    output_filename = FLAGS.corpus_path.split('/')[-1]
    FLAGS.output_filename = '.'.join(output_filename.split('.')[:-1])


def write_to_tf_record(writer, tokenizer, doc_text,
                      docid, doc_type):
  if doc_type=="passage":
    doc_ids = tokenization.convert_to_colbert_input(
        text='[D] '+doc_text, max_seq_length=FLAGS.max_seq_length, tokenizer=tokenizer,
        add_cls=True, padding_mask=False)
  elif doc_type=="query":
    doc_ids = tokenization.convert_to_colbert_input(
        text='[Q] '+doc_text, max_seq_length=FLAGS.max_seq_length, tokenizer=tokenizer,
        add_cls=True, padding_mask=True)

  docid=int(docid)
  doc_ids_tf = tf.train.Feature(
      int64_list=tf.train.Int64List(value=doc_ids))


  docid_tf = tf.train.Feature(
      int64_list=tf.train.Int64List(value=[docid]))


  features = tf.train.Features(feature={
      'doc_ids': doc_ids_tf,
      'docid': docid_tf,
  })
  example = tf.train.Example(features=features)
  writer.write(example.SerializeToString())


def convert_corpus(corpus_path, meta, output_filename, tokenizer, doc_type, offset=0, dummy_line=0, data_format='tsv'):

  corpus = msmarco_corpus(corpus_path, meta, data_format)
  num_lines = corpus.num
  print('{} lines found.'.format(num_lines))

  # remain = num_lines%40
  # if doc_type=='query':
  #   remain = 0
  print('Converting {} to tfrecord...'.format(corpus_path))
  

  counter=0
  id_writer = open(os.path.join(output_filename + '.id'), 'w')
  # if doc_type=='query': # We assume there are not so many query for inference
  writer = tf.python_io.TFRecordWriter(
      os.path.join(output_filename +'.tf'))
  # else:
  #   writer = tf.python_io.TFRecordWriter(
  #       os.path.join(output_filename +'.tf'))

  gen_corpus = corpus.output()
  start_time = time.time()
  i = offset
  for (docid, doc) in tqdm(gen_corpus):

    # if (i % (num_lines-remain) == 0) and (i!=0):
    #   writer.close()
    #   counter+=1
    #   writer = tf.python_io.TFRecordWriter(
    #     os.path.join(output_filename +'.tf'))


    write_to_tf_record(writer=writer,
                       tokenizer=tokenizer,
                       doc_text=doc,
                       docid=i,
                       doc_type=doc_type)

    id_writer.write('%d\t%s\n'%(i,docid))
    i+=1
    # if ((i+1) % max_num_doc) == 0:
    #   print('Writing {} corpus, doc {} of {}'.format(
    #       output_filename, i, num_lines))
    #   time_passed = time.time() - start_time
    #   hours_remaining = (
    #       num_lines - i) * time_passed / (max(1.0, i) * 3600)
    #   print('Estimated hours remaining to write the {} corpus: {}'.format(
    #       output_filename, hours_remaining))
    #   counter+=1
      # writer.close()
      # writer = tf.python_io.TFRecordWriter(
      #   os.path.join(output_filename + '-' + str(counter) +'.tf'))
  if dummy_line>0:
    print('fill {} dummy lines'.format(dummy_line))
    for l in range(dummy_line):
      write_to_tf_record(writer=writer,
                       tokenizer=tokenizer,
                       doc_text='\t',
                       docid=-1,
                       doc_type=doc_type)

      # id_writer.write('%d\t%s\n'%(i,docid))
      # i+=1

  writer.close()
  id_writer.close()




def main():

  print('Loading Tokenizer...')
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=True)

  if not os.path.exists(FLAGS.output_folder):
    os.mkdir(FLAGS.output_folder)


  
  if os.path.isdir(FLAGS.corpus_path):
    files = glob.glob(os.path.join(FLAGS.corpus_path, FLAGS.corpus_prefix + '*'))
  else:
    files = [FLAGS.corpus_path]

  print('Count num of line')
  file_to_line_numer = {}
  for file in files:
    file_to_line_numer[file] = sum(1 for line in open(file))
  
  num_files = len(files)
  if (FLAGS.workers==None) or (FLAGS.workers >= num_files):
    num_workers = num_files
  else:
    num_workers = FLAGS.workers
  if num_workers>1:
    pool = Pool(num_workers)
    num_files_per_worker=num_files//num_workers
    offset = 0
    file_list = []
    for i in range(num_workers):
        f_out = os.path.join(FLAGS.output_folder, FLAGS.output_filename + '-' + str(i)) 
        for file in file_list:
          offset+=file_to_line_numer[file]

        if i==(num_workers-1): #last thread
            file_list = files[i*num_files_per_worker:]
            total_line_number = 0
            for file in file_list:
              total_line_number += file_to_line_numer[file]
            dummy_line = FLAGS.batch_size - total_line_number%(FLAGS.batch_size)
        else:
            file_list = files[i*num_files_per_worker:((i+1)*num_files_per_worker)]
            dummy_line = 0

        pool.apply_async(convert_corpus ,(file_list, FLAGS.meta, f_out, tokenizer, FLAGS.doc_type, offset, dummy_line, FLAGS.data_format))
    pool.close()
    pool.join()
    # f_out = os.path.join(FLAGS.output_folder, FLAGS.output_filename) 
    # convert_corpus(corpus_path=files, meta=FLAGS.meta, output_filename=f_out, tokenizer=tokenizer, doc_type=FLAGS.doc_type)
  else:
    if FLAGS.doc_type=='query':
      dummy_line=0
    else:
      dummy_line = FLAGS.batch_size - total_line_number%(FLAGS.batch_size)
    f_out = os.path.join(FLAGS.output_folder, FLAGS.output_filename) 
    convert_corpus(corpus_path=files, meta=FLAGS.meta, output_filename=f_out, tokenizer=tokenizer, doc_type=FLAGS.doc_type, offset=0, dummy_line=dummy_line, data_format=FLAGS.data_format)
  print('Done!')


if __name__ == '__main__':
  main()
