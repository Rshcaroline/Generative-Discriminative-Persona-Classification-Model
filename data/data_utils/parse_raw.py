import re
import string
import nltk

############################################
# process the raw corpus to the csv format
# remain the main six characters' dialogs
#


file_read = "/Users/aaronhe/FDU/NLP/Generative-Persona-Learning-Model-indev/data/TBBT/test0.1/test0.1.txt"
file_write = "./data/bt_aug_test.csv"
f_read = open(file_read, 'r')
f_write = open(file_write, 'w')

lines = f_read.readlines()
line_write = []

def check_legal(line):
    legal_ch = set(["Sheldon", "Leonard", "Penny", "Howard", "Raj", "Amy"])
    if line.find(": ") != -1:
        ch, sen = string.split(line, ": ", maxsplit=1)
        print ch, sen
        sen = nltk.tokenize.word_tokenize(sen.decode('utf-8'))
        words = ""
        for word in sen:
            words += word + " "
        if ch.strip() in legal_ch:
            return (ch.strip() + "," + "\"" + words + "\",\n").encode('utf-8')
        elif ch.strip() != "" and ch.strip() != "Scene":
            return ("Others" + "," + "\"" + words + "\",\n").encode('utf-8')

    return False


for line in lines:
    new_line = re.sub(r'\(.*\)', '', line)
    legal_line = check_legal(new_line)
    if legal_line:
        legal_line.replace("\"\"", "\"uh . \"")
        line_write.append(legal_line)

f_write.writelines(line_write)

print lines[1]