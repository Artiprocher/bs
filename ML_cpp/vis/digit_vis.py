from PIL import Image
import os
import sys


def save_image(name):
    f = list(map(float, open("vis/data.txt").read().split()))
    mi, ma = min(f), max(f)
    ls = open("vis/data.txt").read().split("\n")[0:-1]
    data = []
    for l in ls:
        data.append(list(map(float, l.split())))
    h, w = len(data), len(data[0])
    img = Image.new("RGB", (h*10, w*10))
    for i in range(h):
        for j in range(w):
            val = data[i][j]
            val = int((val-mi)/(ma-mi)*255)
            for k1 in range(i*10, i*10+10):
                for k2 in range(j*10, j*10+10):
                    img.putpixel((k2, k1), (val, val, val, 0))
    img.save("vis/%s.jpg" % name)


save_image(sys.argv[1])
