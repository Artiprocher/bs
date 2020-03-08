from PIL import Image
import os
img = Image.new("RGB", (28*10, 28*10))
f = list(map(float, open("vis/data.txt").read().split()[0:]))
for i in range(28):
    for j in range(28):
        val = f[i*28+j]
        val = int((val+1.0)*0.5*255)
        for k1 in range(i*10, i*10+10):
            for k2 in range(j*10, j*10+10):
                img.putpixel((k2, k1), (val, val, val, 0))
#img.show()
img.save("vis/GAN%03d.jpg" % (len(os.listdir("vis"))-1))
