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
    article.write(f.read())
    article.write('\n\n')
    f.close()
article.close()