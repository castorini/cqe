"""
This code converts MS MARCO train, dev and eval tsv data into the tfrecord files
that will be consumed by BERT.
"""
import collections
import os
import re
import tensorflow as tf
import time
# local module
import tokenization
# import spacy; from spacy.lang.en import English; nlp = English()
# nlp.add_pipe(nlp.create_pipe('sentencizer'))


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
    "corpus",
    "msmarco",
    "Path to the MSMARCO training dataset containing the tab separated "
    "<query, positive_paragraph, negative_paragraph> tuples.")

flags.DEFINE_string(
    "corpus_path",
    "./data/top1000.dev.tsv",
    "Path to the MSMARCO training dataset containing the tab separated "
    "<query, positive_paragraph, negative_paragraph> tuples.")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string(
    "doc_type", "doc",
    "Denote the type of document, either doc or query")


flags.DEFINE_integer(
    "num_eval_docs", 1000,
    "The maximum number of docs per query for dev and eval sets.")


def write_to_tf_record(writer, tokenizer, doc_text,
                      docid, doc_type):
  # doc = tokenization.convert_to_unicode(doc)
  # doc_token_ids = tokenization.convert_to_bert_input(
  #     text=doc, max_seq_length=FLAGS.max_seq_length, tokenizer=tokenizer,
  #     add_cls=True)

  # docs=[]
  # try:
  #   for doc in nlp(doc_text).sents:
  #     docs.append(doc)
  # except:
  #   print(doc_text+ ' cannot be sentenized')
  #   docs.append(doc_text)

  # doc_token_ids, sent_segment, start_pos, sent_num = tokenization.convert_seqs_to_bert_input(
  #       docs=docs,
  #       max_seq_length=FLAGS.max_seq_length,
  #       tokenizer=tokenizer,
  #       add_cls=True)

  if doc_type=="doc":
    doc_ids, _, _ = tokenization.convert_to_colbert_input(
        text='[D] '+doc_text, max_seq_length=FLAGS.max_seq_length, tokenizer=tokenizer,
        add_cls=True, padding_mask=False)
  elif doc_type=="query":
    doc_ids, _, _ = tokenization.convert_to_colbert_input(
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


def convert_corpus(corpus, tokenizer, doc_type):
  print('Converting {} to tfrecord...'.format(FLAGS.corpus_path))
  start_time = time.time()
  docids=[]
  docs=[]

  print('Counting number of examples...')
  num_lines = sum(1 for line in open(FLAGS.corpus_path, 'r'))
  print('{} examples found.'.format(num_lines))
  if num_lines > 1000000:
    num_lines0 = num_lines - num_lines%40
  else:
    num_lines0 = num_lines + 1

  id_writer = open(FLAGS.output_folder + '/'+ corpus + '.id', 'w')
  with open(FLAGS.corpus_path) as f:
      for line in f:
        try:
          docid, doc = line.strip().split('\t')

        except:
          doc=[]
          for i, content in enumerate(line.strip().split('\t')):
            if i==0:
              docid=content
            else:
              doc.append(content)

          doc=' '.join(doc)
        docids.append(docid)
        docs.append(doc)

  # if set_name == 'dev':
  #   dataset_path = FLAGS.dev_dataset_path
  #   relevant_pairs = set()
  #   with open(FLAGS.dev_qrels_path) as f:
  #     for line in f:
  #       query_id, _, doc_id, _ = line.strip().split('\t')
  #       relevant_pairs.add('\t'.join([query_id, doc_id]))
  # else:
  #   dataset_path = FLAGS.eval_dataset_path

  # queries_docs = collections.defaultdict(list)
  # query_ids = {}
  # with open(dataset_path, 'r') as f:
  #   for i, line in enumerate(f):
  #     query_id, doc_id, query, doc = line.strip().split('\t')
  #     label = 0
  #     if set_name == 'dev':
  #       if '\t'.join([query_id, doc_id]) in relevant_pairs:
  #         label = 1
  #     queries_docs[query].append((doc_id, doc, label))
  #     query_ids[query] = query_id

  # # Add fake paragraphs to the queries that have less than FLAGS.num_eval_docs.
  # queries = list(queries_docs.keys())  # Need to copy keys before iterating.
  # for query in queries:
  #   docs = queries_docs[query]
  #   docs += max(
  #       0, FLAGS.num_eval_docs - len(docs)) * [('00000000', 'FAKE DOCUMENT', 0)]
  #   queries_docs[query] = docs

  # assert len(
  #     set(len(docs) == FLAGS.num_eval_docs for docs in queries_docs.values())) == 1, (
  #         'Not all queries have {} docs'.format(FLAGS.num_eval_docs))
  
  counter=0
  writer = tf.python_io.TFRecordWriter(
      FLAGS.output_folder + '/'+ corpus + str(counter) +'.tf')

  for i, doc in enumerate(docs):

    write_to_tf_record(writer=writer,
                       tokenizer=tokenizer,
                       doc_text=doc,
                       docid=i,
                       doc_type=doc_type)

    id_writer.write('%d\t%s\n'%(i,docids[i]))
    if (i+1) % 1000000 == 0:
      print('Writing {} corpus, doc {} of {}'.format(
          corpus, i, len(docs)))
      time_passed = time.time() - start_time
      hours_remaining = (
          len(docs) - i) * time_passed / (max(1.0, i) * 3600)
      print('Estimated hours remaining to write the {} corpus: {}'.format(
          corpus, hours_remaining))
    if (i+1) % num_lines0 == 0:
      writer.close()
      counter+=1
      writer = tf.python_io.TFRecordWriter(
        FLAGS.output_folder + '/'+ corpus + str(counter) +'.tf')

  id_writer.close()
  writer.close()




def main():

  print('Loading Tokenizer...')
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=True)

  if not os.path.exists(FLAGS.output_folder):
    os.mkdir(FLAGS.output_folder)


  convert_corpus(corpus=FLAGS.corpus, tokenizer=tokenizer, doc_type=FLAGS.doc_type)
  print('Done!')

if __name__ == '__main__':
  main()
