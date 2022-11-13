
from operator import mod
import cv2
import numpy as np
import math
import random

img = "opencv.jpg"
msg = "shiq"
k_location = []

print("PVD ENCODING: ")
'''读取灰度图像'''
coverImg = cv2.imread(img)
origImg = coverImg
coverImg = coverImg[:,:,0]
print("image size: ", coverImg.shape)

'''将字符串转换为二进制序列'''
secretMsg = ''.join(map(bin, bytearray(msg, 'utf8')))
secretMsg = secretMsg.replace("0b", "")
msgLenth = len(secretMsg)
print(secretMsg)
print(type(secretMsg))

height, width = coverImg.shape
msgcount = 0

r = 0
while r <= 8:
    i = random.randint(1,height-1)
    j = random.randint(1,width-1)
    print("i: ",i,"j: ",j)
    k = (i, j)
    p = np.array([coverImg[i,j-1], coverImg[i-1,j-1], coverImg[i-1,j], coverImg[i-1,j+1], coverImg[i,j], \
                coverImg[i,j+1], coverImg[i+1,j+1], coverImg[i+1,j], coverImg[i+1,j-1]], dtype=int)
    sortedp = sorted(p)
    pm = sortedp[4]
    D = np.fabs(sortedp-pm)
    D = np.delete(D,4)
    print("sortedp: ",sortedp)
    print("pm: ",pm)
    print("D: ",D)
    dmin = np.min(D)
    dmax = np.max(D)
    alphe = 0.6
    r = alphe * (dmax - dmin) + dmin
print("-----------------------")
print("p: ", p)
print("r: ", r)
print("k: ", k)
nbits = max(int(math.log2(r)), 1)
bitsnum = 9 * nbits
print("nbits: ", nbits)
print(str(bitsnum) + " bits can be enbeded")
pixelnum = math.ceil(msgLenth/nbits)
print(str(pixelnum) + " pixel are/is needed")

po = [(i, j-1),(i-1, j-1),(i-1,j),(i-1,j+1),(i,j),(i,j+1),(i+1,j+1),(i+1,j),(i+1,j-1)]

for index in range(0, pixelnum):

    extractedmsg = secretMsg[msgcount:msgcount + nbits]
    msgcount = msgcount + nbits
    print("extractedmsg: ",extractedmsg)
    extractedmsg1 = extractedmsg[::-1]
    print("msgcount", msgcount)
    h = int(extractedmsg1, 2)
    print("h: ",h)
    '''    k_location.append(k)
    print("k_location: ", k_location)'''
    print("original pixel value: ", coverImg[po[index]])
    coverImg[po[index]] = p[index] - mod(p[index], 2**nbits) + h
    print("changed pixel value: ", coverImg[po[index]])
    print("-----------------------")


print("----------------------------------------------------------------------------------------------\n")
print("PVD DECODING: nbits and location k are given")
s2 = ''
bitcount = 0
for index1 in range(0, pixelnum):
    h = mod(coverImg[po[index1]], 2**nbits)
    print(h)
    s = bin(h)
    s = s.replace("0b", "")
    bitcount = bitcount + nbits
    if bitcount < msgLenth:
        s = s.zfill(nbits)
        s1 = s[::-1]
    else:
        s = s.zfill(msgLenth-bitcount)
        s1 = s[::-1]
    
    s2 = s2 + s1
    print(s2)
print(type(s2))
s2length = len(s2)
print(type(s2length))
s22 = int(s2length/7)
co = 0
exMsg = ""
for index2 in range(0, s22):
    asciicode = int(s2[7*index2:7*(index2+1)], 2)
    print("asciicode: ", asciicode)
    msg = chr(asciicode)
    exMsg = exMsg + msg
  
print(exMsg)

print(coverImg.shape)
cv2.imshow("withmsg", coverImg)
cv2.imshow("nomsg", origImg[:, :, 0])
cv2.waitKey()


