from PIL import Image
import os

path=r"E:\IndoorDataset\Images"
output_file=open(path+"\\"+"data.csv","w")
label_list=["airport_inside","bar","bedroom","kitchen","livingroom"]
W,H=10,10
for i in label_list:
    image_list=os.listdir(path+"\\"+i)
    print(i,len(image_list))
    for j in image_list:
        print(j)
        img=Image.open(path+"\\"+i+"\\"+j)
        img=img.resize((W,H),Image.ANTIALIAS)
        if type(img.getpixel((0,0)))!=tuple:
            continue
        for k in range(3):
            for h in range(H):
                for w in range(W):
                    output_file.write(str(img.getpixel((w,h))[k]))
                    output_file.write(",")
        output_file.write(i+"\n")
output_file.close()
