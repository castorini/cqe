import argparse
from pyserini.search import SimpleSearcher
from progressbar import *
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='output corpus from pyserini prebuilt index')
    parser.add_argument('--output_folder', required=True, help='output folder')
    parser.add_argument('--index', required=True, help='index path')
    args = parser.parse_args()
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    fout = open(os.path.join(args.output_folder, 'collection.tsv'), 'w')
    index = SimpleSearcher.from_prebuilt_index(args.index)
    num_of_doc = index.num_docs
    widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),
               ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets, maxval=10*num_of_doc).start()
    for i in range(num_of_doc):
        docid = index.doc(i).docid()
        content = index.doc(i).raw()
        fout.write('{}\t{}\n'.format(docid, content))
        pbar.update(10 * i + 1)