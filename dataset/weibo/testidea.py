
from collections import defaultdict


userdic = defaultdict(list)
with open("./weibo.train", 'r', encoding='utf8') as fin:
    for line in fin.readlines():
        arr = line.strip().split()
        userdic[arr[0]].append(arr[-1])


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

print(userdic)