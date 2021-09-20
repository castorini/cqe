import os
import glob
import json
class msmarco_corpus():
	def __init__(self, files, meta, data_format):
		self.meta = meta.split(',')
		self.data_format = data_format
		self.files = files
		print('Total {} files'.format(len(self.files)))
		self.lines = []
		for file in self.files:
			with open(file, 'r') as f:
				self.lines+=f.readlines()
		self.num = len(self.lines)
	def output(self):
		for i, line in enumerate(self.lines):
			if self.data_format=='tsv': #tsv format id \t text
				info = line.strip().split('\t')
				docid = info[0]
				text = ' '.join(info[1:])
			elif self.data_format=='json':  #json format
				info = json.loads(line.strip())
				docid = info['id']
				# if (self.meta[0]=='contents') and (len(self.meta)==1): #msmarcov1 doc json format
				# 	text = info['contents']
				# 	fields = text.split('\n')
				# 	title, text = fields[1], fields[2:]
				# 	if len(text) > 1:
				# 		text = ' '.join(text)
				# 		text = title + ' ' + text
				# else:
				text = []
				for meta in self.meta:
					text.append(info[meta])
				text = ' '.join(text)

			yield docid, text