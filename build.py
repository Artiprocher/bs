import os
import sys

file_list = os.listdir()
print(file_list)
chapter_num = 0
for i in file_list:
    if i[:4] == 'part':
        chapter_num += 1
article = open('article.md', encoding = 'utf-8', mode = 'w')
for i in range(chapter_num):
    f = open('./part%02d/part%02d.md' % (i, i), encoding = 'utf-8')
    code_flag = 0
    for line in f:
        if line == '```file\n':
            code_flag = 1
        elif code_flag == 1:
            code = open(line.strip())
            article.write('```cpp\n' + code.read().strip() + '\n```\n')
            code.close()
            code_flag = 2
        elif code_flag == 2:
            code_flag = 0
            continue
        else:
            article.write(line)
    article.write('\n\n')
    f.close()
article.close()
os.system('pandoc -o article.docx article.md')