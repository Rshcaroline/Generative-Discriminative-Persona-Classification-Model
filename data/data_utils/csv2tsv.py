import os
import argparse
import nltk
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--path', default='../data/',
                    help='Input data path.')
parser.add_argument('--save_path', default='../../data/K_data/',
                    help='Output data path.')
parser.add_argument('--prefix', default='bt_aug_',
                    help='prefix for data')
parser.add_argument('--k', type=int, default=4,
                    help='Number of sentences in context')

args = parser.parse_args()


def split():
    num_sentence = args.k + 1
    for data in ("train", "test", "dev"):
        output = open(os.path.join(args.save_path, args.prefix+data+str(num_sentence)+".csv"), 'w')
        corpus = pd.read_csv(os.path.join(args.path, args.prefix+data+".csv"), header=None)
        dialog = corpus[[2 * x + 1 for x in range(num_sentence / 2 + 1)]]
        speaker = corpus[2*num_sentence-2]
        dialog += '</s>'
        for ch,sen in zip(speaker.index, dialog.index):
            for i in range(args.k):
                print >> output, dialog.loc[sen].iloc[i],
            print >> output, "\t"+speaker.loc[ch]+"\t"+dialog.loc[sen].iloc[i]

if __name__ == "__main__":
    split()
