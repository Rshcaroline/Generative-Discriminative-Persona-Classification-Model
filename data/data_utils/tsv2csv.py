import pandas as pd
import string

filename = './dev0.1.tsv'
output = "./Au2Flower.csv.dev"

out = open(output, 'w')

spks = set(["Sheldon","Howard","Penny","Raj","Leonard","Amy"])

with open(filename, 'r') as f:
    lines = f.readlines()
    res = [line.split('\t') for line in lines]
    for context, speaker, response in res:  # res[26009] {tab} for 5 times
        if speaker in spks:
            context, response = context.replace('</s>', ''), response.replace('</s>', '')
            string = "Fake,"+"\"" + context + "\"," + speaker + ",\"" + response[:-1] + "\","
            print >> out, string