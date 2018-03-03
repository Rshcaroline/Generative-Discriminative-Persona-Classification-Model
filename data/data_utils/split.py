import os

files = os.listdir("./raw_corpus")
for file in files:
    with open(os.path.join('./raw_corpus',file),'r') as f:
        lines = f.readlines()
        train = open(os.path.join('./train0.8', file), 'w')
        dev = open(os.path.join('./dev0.1', file), 'w')
        test = open(os.path.join('./test0.1', file), 'w')
        
        length = len(lines)

        for line in [lines[i] for i in range(int(0.8*length))]:
            print >> train, line,
        for line in  [lines[i] for i in range(int(0.8*length), int(0.9*length))]:
            print >> dev, line,
        for line in [lines[i] for i in range(int(0.9*length), length)]:
            print >> test, line,
