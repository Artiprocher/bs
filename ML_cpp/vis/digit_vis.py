from PIL import Image
import os
import sys


def get_color(val):
    c1 = (0.0, 0.4, 0.8, 0.0)
    c2 = (0.9, 0.6, 0.0, 0.0)
    c = tuple(int((c1[i]+(c2[i]-c1[i])*val)*255) for i in range(4))
    return c


def save_image(name):
    f = list(map(float, open("vis/data.txt").read().split()))
    mi, ma = min(f), max(f)
    ls = open("vis/data.txt").read().split("\n")[0:-1]
    data = []
    for l in ls:
        data.append(list(map(float, l.split())))
    h, w = len(data), len(data[0])
    d = 30
    img = Image.new("RGB", (h*d, w*d))
    for i in range(h):
        for j in range(w):
            val = data[i][j]
            val = (val-mi)/(ma-mi)
            for k1 in range(i*d, i*d+d):
                for k2 in range(j*d, j*d+d):
                    img.putpixel((k2, k1), get_color(val))
    img.save("vis/%s.jpg" % name)


save_image(sys.argv[1])
