import pandas
from collections import deque

##### slide window to enlarge data
# eg. context: 1 2 3 respone: 4 ; context: 5 6 7 respone: 8
# = > context: 1 2 3 respone: 4 ; context: 2 3 4 respone: 5 ; context: 3 4 5 respone: 6 ...
K = 4

for data in ("train", "dev", "test"):
    INPUT_PATH = "/Users/aaronhe/FDU/NLP/Generative-Persona-Learning-Model-indev/shran/data/bt_aug_"+data+".csv"
    OUTPUT_PATH = "/Users/aaronhe/FDU/NLP/Generative-Persona-Learning-Model-indev/shran/data/bt_aug_"+data+str(K)+".csv"

    f_dst = open(OUTPUT_PATH, 'w')

    q = deque([])

    with open(INPUT_PATH, 'r') as f:
        src_lines = f.readlines()

        for i, line in enumerate(src_lines):
            q.append(line[:-1])
            if len(q) == K:
                for line in q:
                    print >> f_dst, line,
                    print line,
                print ''
                print >> f_dst, ''
                q.popleft()

