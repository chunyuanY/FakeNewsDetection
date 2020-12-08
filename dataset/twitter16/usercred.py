
from collections import defaultdict


userdic = defaultdict(list)
with open("./twitter16.train", 'r', encoding='utf8') as fin:
    for line in fin.readlines():
        arr = line.strip().split("\t")
        userdic[arr[0]].append(arr[-1])
        with open("./tree/"+str(arr[1])+".txt", 'r', encoding='utf8') as fin2:
            fin2.readline()
            for lin in fin2.readlines():
                rep = eval(lin.strip().split("->")[1])
                userdic[rep[0]].append(arr[-1])


for k, v in userdic.items():
    positive = 0
    negative = 0

    for label in v:
        if label in ['non-rumor', 'true']:
            positive += 1
        else:
            negative += 1

    ulabel = -1
    if positive == 0:
        ulabel = 2
    elif negative == 0:
        ulabel = 0
    else:
        ulabel = 1
    userdic[k] = ulabel


with open("user_credibility.txt", 'w', encoding='utf8') as fout:
    fout.write(str(dict(userdic)))


