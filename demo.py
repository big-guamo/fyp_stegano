from operator import  mod
import re
from select import select
import cv2
import numpy as np
import math
import random
import metrics

class PvdEncoding:

    def __init__(self, img, msg):
        self.img = img
        self.msg = msg
    
    def msgToBinary(self):
        '''to obtain the binary form of secret message'''
        secretMsg = ''.join(map(bin, bytearray(self.msg, 'utf8')))
        secretMsg = secretMsg.replace("0b", "")
        return secretMsg
    
    def readImage(self):
        '''read the original image and its blue channel'''
        coverImg = cv2.imread(self.img)
        origImg = coverImg
        coverImg = coverImg[:,:,0]
        return (coverImg, origImg)

    def ﬂuctuationRange(self):
        '''to randomly select the pixel block and obtain the position K of the block 
           and the n bits each pixel can be embeded'''
        r = 0
        coverImg = self.readImage()
        coverImg = coverImg[0]
        height, width = coverImg.shape
        while r <= 8:
            #i = random.randint(1,height-1)
            #j = random.randint(1,width-1)
            i = 184
            j = 52
            print("i: ",i,"j: ",j)
            k = (i, j)
            '''p is the block with 9 pixels in it'''
            p = np.array([coverImg[i,j-1], coverImg[i-1,j-1], coverImg[i-1,j], coverImg[i-1,j+1], coverImg[i,j], \
                        coverImg[i,j+1], coverImg[i+1,j+1], coverImg[i+1,j], coverImg[i+1,j-1]], dtype=int)
            sortedp = sorted(p)
            pm = sortedp[4]
            '''store the difference between p and pm into the vector D'''
            D = np.fabs(sortedp-pm)
            D = np.delete(D,4)
            print("sortedp: ",sortedp)
            print("pm: ",pm)
            print("D: ",D)
            dmin = np.min(D)
            dmax = np.max(D)
            alphe = 1
            r = alphe * (dmax - dmin) + dmin
        print("-----------------------")
        print("p: ", p)
        print("r: ", r)
        print("k: ", k)
        nbits = max(math.ceil((math.log2(r))), 1)
        bitsnum = 9 * nbits
        print("nbits: ", nbits)
        print(str(bitsnum) + " bits can be enbeded")
        
        return (k, nbits)

    def encoding(self):
        '''write the secret message into the cover image'''
        secretMsg = self.msgToBinary()
        print(secretMsg)
        print(type(secretMsg))
        print("msgLength: ", len(secretMsg))
        image = self.readImage()
        coverImg = image[0]
        
        I = self.fluctuationRange()
        k = I[0]
        nbits = I[1]
        i, j = k
        p = np.array([coverImg[i,j-1], coverImg[i-1,j-1], coverImg[i-1,j], coverImg[i-1,j+1], coverImg[i,j], \
                coverImg[i,j+1], coverImg[i+1,j+1], coverImg[i+1,j], coverImg[i+1,j-1]], dtype=int)
        po = [(i, j-1),(i-1, j-1),(i-1,j),(i-1,j+1),(i,j),(i,j+1),(i+1,j+1),(i+1,j),(i+1,j-1)]
        msgcount = 0
        pixelnum = math.ceil(len(secretMsg)/nbits)
        for index in range(0, pixelnum):
            extractedmsg = secretMsg[msgcount:msgcount + nbits]
            msgcount = msgcount + nbits
            print("extractedmsg: ",extractedmsg)
            extractedmsg1 = extractedmsg[::-1]
            print("msgcount", msgcount)
            h = int(extractedmsg1, 2)
            print("h: ",h)
            print("original pixel value: ", coverImg[po[index]])
            coverImg[po[index]] = p[index] - mod(p[index], 2**nbits) + h
            print("changed pixel value: ", coverImg[po[index]])
            print("-----------------------")

        return (coverImg, k, nbits)


class PvdDecoding:

    def __init__(self, img, nbits, k, len):
        self.img = img
        self.nbits = nbits
        self.k = k
        self.len = len

    def decoding(self):
        i, j = self.k
        po = [(i, j-1),(i-1, j-1),(i-1,j),(i-1,j+1),(i,j),(i,j+1),(i+1,j+1),(i+1,j),(i+1,j-1)]
        pixelnum = math.ceil(self.len/self.nbits)
        s2 = ''
        bitcount = 0
        for index1 in range(0, pixelnum):   
            h = mod(self.img[po[index1]], 2**self.nbits)
            print("h: ", h)
            s = bin(h)
            s = s.replace("0b", "")
            bitcount = bitcount + self.nbits
            if bitcount < self.len:
                s = s.zfill(self.nbits)
                s1 = s[::-1] 
            else:
                s = s.zfill(self.len-bitcount)
                s1 = s[::-1]
            
            s2 = s2 + s1
            print(s2)

        print(type(s2))
        s2length = len(s2)
        s22 = int(s2length/7)
        exMsg = ""
        for index2 in range(0, s22):
            asciicode = int(s2[7*index2:7*(index2+1)], 2)
            print("asciicode: ", asciicode)
            msg = chr(asciicode)
            exMsg = exMsg + msg
        
        return exMsg


img = "suzhou.jpg"
msg = "HELLO!"

'''in embedding process, the secret message and cover image are two parameters'''
P = PvdEncoding(img, msg)
Pe = P.encoding()
image_with_msg = Pe[0]
image_without_msg = P.readImage()
image_without_msg = image_without_msg[1]
print(image_with_msg.shape)
length_ = len(P.msgToBinary())

metrics = metrics.PNSR(image_without_msg[:,:,0], image_with_msg)
print("PNSR: ", metrics.MeanSquareError())

'''----------------------------------------------------------------------------------------------------------'''
def ssim(img1, img2):
  C1 = (0.01 * 255)**2
  C2 = (0.03 * 255)**2
  img1 = img1.astype(np.float64)
  img2 = img2.astype(np.float64)
  kernel = cv2.getGaussianKernel(11, 1.5)
  window = np.outer(kernel, kernel.transpose())
  mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] # valid
  mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
  mu1_sq = mu1**2
  mu2_sq = mu2**2
  mu1_mu2 = mu1 * mu2
  sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
  sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
  sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
  ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                              (sigma1_sq + sigma2_sq + C2))
  return ssim_map.mean()
  
def calculate_ssim(img1, img2):
  '''calculate SSIM
  the same outputs as MATLAB's
  img1, img2: [0, 255]
  '''
  if not img1.shape == img2.shape:
    raise ValueError('Input images must have the same dimensions.')
  if img1.ndim == 2:
    return ssim(img1, img2)
  elif img1.ndim == 3:
    if img1.shape[2] == 3:
      ssims = []
      for i in range(3):
        ssims.append(ssim(img1, img2))
      return np.array(ssims).mean()
    elif img1.shape[2] == 1:
      return ssim(np.squeeze(img1), np.squeeze(img2))
  else:
    raise ValueError('Wrong input image dimensions.')

ss = calculate_ssim(image_without_msg[:,:,0], image_with_msg)
print("SSIM: ",ss)

'''----------------------------------------------------------------------------------------------------------'''
cv2.imshow("image_with_msg", image_with_msg)
cv2.imwrite('opencv_withmsg.jpg',image_with_msg,[int(cv2.IMWRITE_JPEG_QUALITY),70])
cv2.imshow("image_without_msg", image_without_msg[:,:,0])
cv2.waitKey()



'''in extraction process, the n bits and the position k are given'''
'''k = Pe[1]
print(k)  
nbits = Pe[2]
print(nbits)

print(cv2.imread("opencv_withmsg.jpg").shape)
V = PvdDecoding(image_with_msg, nbits, k, length_)
print(V.decoding())'''
'''----------------------------------------------------------------------------------------------------------------'''
