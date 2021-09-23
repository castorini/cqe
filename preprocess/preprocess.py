import json
import argparse
import os
from os.path import join
import random
# from anserini_search import Searcher
# from spacy.lang.en import English
# import tfrecord_generation.tokenization as tokenization
# nlp = English()

# Searcher = Searcher("../index/lucene-index-cast")
# print('Loading Tokenizer...')
# tokenizer = tokenization.FullTokenizer(
#     vocab_file='../uncased_L-12_H-768_A-12/vocab.txt', do_lower_case=True)

def output_conversation_query(qa_path, cqr_path, output):
  """Read a SQuAD json file into a list of SquadExample."""
  qa_files = ['train_v0.2.json', 'val_v0.2.json']
  qa_data = []
  for file in qa_files:
    with open(os.path.join(qa_path, file), "r") as reader:
      qa_data += json.load(reader)["data"]
  cqr_files = ['train.json', 'dev.json', 'test.json']
  cqr_data = []
  for file in  cqr_files:
    with open(os.path.join(cqr_path, file), "r") as reader:
      cqr_data += json.load(reader)


  titles = {}
  descriptions = {}
  qid_to_answer = {}
  for entry in qa_data:
    for paragraph in entry["paragraphs"]: #dict_keys(['paragraphs', 'section_title', 'background', 'title'])
      titles[paragraph['id']] = entry['section_title']
      descriptions[paragraph['id']] = entry['background']
      for qa in paragraph["qas"]:
        qid_to_answer[qa['id']] = qa['orig_answer']

  rw_Q=0
  no_rw_Q=0
  session=0
  conversation_sessions = []
  # with open(os.path.join(output_path, 'train_query_response.tsv'), "w") as writer:
  for cqr in cqr_data:
    if cqr['Question_no']==1:
      if session!=0:
        conversation['turn'] = conversation_turns
        conversation_sessions.append(conversation)
      conversation = {}
      conversation_turns = []
      session+=1
      conversation['number'] = session

      conversation['description'] = descriptions[cqr['QuAC_dialog_id']]
      conversation['title'] = titles[cqr['QuAC_dialog_id']]
      origin_query = cqr['Rewrite']


    else:
      origin_query = cqr['Question'] 
    rewrite_query = cqr['Rewrite']



    if 'interesting aspects' not in origin_query:
      try:
        conversation_turns.append({'number': cqr['Question_no'], 'raw_utterance':origin_query, "manual_rewritten_utterance":rewrite_query, 'canonical_result':qid_to_answer[cqr['QuAC_dialog_id']+'_q#'+str(cqr['Question_no']-1)]['text'] })
      except:
        conversation_turns.append({'number': cqr['Question_no'], 'raw_utterance':origin_query, "manual_rewritten_utterance":rewrite_query, 'canonical_result':'no answer' })
        print('no answer')


      writer.write('{}\t{}\n'.format(str(session)+'_'+str(cqr['Question_no']), rewrite_query))



    with open(os.path.join(output), 'w') as writer:
      data = json.dumps(conversation_sessions)
      writer.write(data)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--QA_path", required=True)
    parser.add_argument("--CQR_path", required=True)
    parser.add_argument("--output", required=True)


    args = parser.parse_args()



    output_conversation_query(args.QA_path, args.CQR_path, args.output)




if __name__ == "__main__":
    main()