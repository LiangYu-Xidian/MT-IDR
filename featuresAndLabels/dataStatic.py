def counta(s):
    count0 = 0
    count1 = 0
    for i in range(len(s)):
        if(s[i] == '0'):
           count0 = count0 + 1
        elif(s[i] == '1'):
           count1 = count1 + 1
    return count0, count1

count0_all = 0
count1_all = 0

with open("train_label_morfs.fasta", encoding='utf-8') as f:
    lines = f.readlines()
    for i in range(0, len(lines), 2):
        label = lines[i+1]
        count0, count1 = counta(label)
        count0_all = count0_all + count0
        count1_all = count1_all + count1

print(count0_all)
print(count1_all)
