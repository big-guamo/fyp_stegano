import numpy as np
import cv2

img = cv2.imread("lena.jpeg")
print(img.shape)
msg = "RR"
key_matrix = np.array([[0,0,0],[0,1,0],[1,1,0]])
char_matrix = np.array([1,1,1,1,1,1,1,1,1])

for i in range(len(msg)):
    char = msg[i]
    char = ord(char)
    char = bin(char).replace("0b", "").zfill(9)
    
    
    for j in range(len(char)):
        char_matrix[j] = char[j]
        char_matrix2 = char_matrix.reshape(3,3)
        char_matrix2 = np.rot90(char_matrix2, -1)
    print(char_matrix2)

T = key_matrix^char_matrix2
T = T.ravel().ravel()
print(T)
print(key_matrix^char_matrix2)

