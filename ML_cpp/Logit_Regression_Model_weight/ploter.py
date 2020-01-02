from PIL import Image
img = Image.new("RGB",(28*10*5,28*10*2))
for number in range(10):
    if number<5:
        dx=number*28*10
        dy=0
    else:
        dx=(number-5)*28*10
        dy=28*10
    f=list(map(float,open("L%d.ini" % number).read().split()[1:]))
    for i in range(28):
        for j in range(28):
            val=f[i*28+j]
            val=int((val-min(f))/(max(f)-min(f))*255)
            for k1 in range(i*10,i*10+10):
                for k2 in range(j*10,j*10+10):
                    img.putpixel((dx+k2,dy+k1),(val,val,val,0))
img.show()