import threading
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import tkinter as tk

##### Input trainned model #####
model = keras.models.load_model('mnist_model1.h5')
model2 = keras.models.load_model('alpha_model.h5')

def find_license(image_name):
    imagepath = str(image_name)
    image = cv2.imread(imagepath)
    RGB_image = image[:,:,::-1]
    gray = cv2.cvtColor(RGB_image, cv2.COLOR_RGB2GRAY)
    low_threshold = 100
    high_threshold = 200
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    '''gaussianblur'''
    kernel_size = 3
    img = cv2.GaussianBlur(edges,(kernel_size, kernel_size), 0)

    '''make the color of the image including only white and black'''
    for i in range (len(img)):
        for j in range (len(img[0])):
            if img[i][j] > 0:
                img[i][j] = 255

    '''the kernal to dilate and erode'''
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    element3 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    element4 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 10))
    element5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    '''use open and close'''
    img = cv2.dilate(img, element1, iterations = 2)
    img = cv2.erode(img, element1, iterations = 2)
    img = cv2.erode(img, element2, iterations = 10)
    img = cv2.erode(img, element5, iterations = 5)
    img = cv2.dilate(img, element5, iterations = 5)
    img = cv2.dilate(img, element2, iterations = 10)
    img = cv2.erode(img, element3, iterations = 10)
    img = cv2.dilate(img, element3, iterations = 10)
    img = cv2.erode(img, element4, iterations = 5)
    img = cv2.dilate(img, element4, iterations = 5)

    '''get the license plate'''
    img_final = np.zeros((len(gray),len(gray[0]),3),np.uint32)
    for i in range (len(img_final)):
        for j in range (len(img_final[0])):
            if img[i][j] == 255:
                img_final[i][j] = RGB_image[i][j]
            else:
                img_final[i][j] = 0

    '''get the coordinate of license plate'''
    i_site = []
    j_site = []
    for i in range (len(img_final)):
        for j in range (len(img_final[0])):
            if img_final[i][j][0] != 0:
                i_site.append(i)
                j_site.append(j)
    i_first = i_site[0]
    j_first = j_site[0]
    i_last = i_site[-1]
    j_last = j_site[-1]

    '''print the license plate'''
    card = img_final[i_first:i_last,j_first:j_last]
    card_gray = cv2.cvtColor(np.float32(card), cv2.COLOR_RGB2GRAY)
    ret,card_bw = cv2.threshold(card_gray,127,255,cv2.THRESH_BINARY)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    card_bw = cv2.dilate(card_bw, element, iterations = 2)
    card_bw = cv2.erode(card_bw, element, iterations = 2)

    '''erase the part of all white above'''
    i_allzero = []
    for i in range (len(card)):
        count = 0
        for j in range (len(card_bw[0])):
            if card_bw[i][j] == 0:
                count += 1
        i_allzero.append(count)            
    i_probable = []
    for i in range (len(i_allzero)):
        if (int(i_allzero[i])) == 0:
            i_probable.append(i)

    '''cut the white space'''
    i_min = 0
    i_max = 0
    max_count = 0
    for i in range (len(i_probable)-1):
        count = i_probable[i+1] - i_probable[i]
        if count > max_count:
            max_count = count
            i_min = i_probable[i]
            i_max = i_probable[i+1]
    card_bw = card_bw[i_min:i_max,:]

    '''find the white part of all i'''
    j_allzero = []
    for i in range (len(card_bw[0])):
        count = 0
        for j in range (len(card_bw)):
            if card_bw[j][i] == 0:
                count += 1
        j_allzero.append(count)    
    j_probable = []
    for i in range (len(j_allzero)):
        if (int(j_allzero[i])) == 0:
            j_probable.append(i)

    '''get the word of license plate'''
    j_final = []
    for i in range (len(j_probable)-1):
        if j_probable[i+1] - j_probable[i] > 1:
            j_final.append(j_probable[i])
    j_final.append(j_probable[-1])

    '''find dash,english word,number'''
    value = []
    for i in range (len(j_final)-1):
        value.append(j_final[i+1]-j_final[i])
    dash = min(value)
    index = 0
    for i in range (len(value)):
        if value[i] == dash:
            dash_index = j_final[i:i+2]
            index = i
    length1 = j_final[index] - j_final[0]
    length2 = j_final[-1] - j_final[index+1]
    if length2 > length1:
        if len(j_final) == 9:
            card_eng = card_bw[:,j_final[0]:j_final[3]]
            card_dash = card_bw[:,j_final[3]:j_final[4]]
            card_num = card_bw[:,j_final[4]:]

    '''get the number and word of license plate'''
    num_1 = card_num[:,0:int((1/4)*len(card_num[0]))]
    num_2 = card_num[:,int((1/4)*len(card_num[0])):int((2/4)*len(card_num[0]))]
    num_3 = card_num[:,int((2/4)*len(card_num[0])):int((3/4)*len(card_num[0]))]
    num_4 = card_num[:,int((3/4)*len(card_num[0])):len(card_num[0])]

    alphabet_1 = card_eng[:,0:int((1/3)*len(card_eng[0]))]
    alphabet_2 = card_eng[:,int((1/3)*len(card_eng[0])):int((2/3)*len(card_eng[0]))]
    alphabet_3 = card_eng[:,int((2/3)*len(card_eng[0])):len(card_eng[0])]

    ''' save the numbers and alphabet'''
    plt.imshow(num_1)
    cv2.imwrite('num_1.jpg', num_1)
    plt.imshow(num_2)
    cv2.imwrite('num_2.jpg', num_2)
    plt.imshow(num_3)
    cv2.imwrite('num_3.jpg', num_3)
    plt.imshow(num_4)
    cv2.imwrite('num_4.jpg', num_4)
    plt.imshow(alphabet_1)
    plt.savefig('alphabet_1.png')
    plt.imshow(alphabet_2)
    plt.savefig('alphabet_2.png')
    plt.imshow(alphabet_3)
    plt.savefig('alphabet_3.png')

    '''input the number into CNN to predict'''
    def test_img(imagepath):
        image = cv2.imread(imagepath)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        ret,thresh1 = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
        img = cv2.resize(image,(28,28))
        img = 255 - img
        im_array = np.array(img)
        im_array = np.reshape(im_array, (1,28,28,1))
        im_array = im_array.astype('float32')/255
        predict = model.predict_classes(im_array)
        ans.append(predict[0])

    def test_alphabet(imagepath):
        alphabet = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
        element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        image = cv2.imread(imagepath)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        image = cv2.dilate(image, element1, iterations = 5)
        img = cv2.resize(image,(28,28))
        img = 255 - img
        for i in range(len(img)):
            for j in range(len(img[0])):
                if img[i][j] < 100:
                    img[i][j] = 0
        im_array = np.array(img)
        im_array = np.reshape(im_array, (1,28,28,1))
        im_array = im_array.astype('float32')/255
        predict = model2.predict_classes(im_array)
        ans.append(str(alphabet[predict[0]]))

    '''using thread to increase rcognition speed'''
    global ans
    ans = []
    alps = ['alphabet_1.png','alphabet_2.png','alphabet_3.png']
    nums = ['num_1.jpg','num_2.jpg','num_3.jpg','num_4.jpg']
    threads = []
    for i in range(7):
        if i <= 2:
            threads.append(threading.Thread(target = test_alphabet(alps[i])))
            threads[i].start() 
        else:  
            threads.append(threading.Thread(target = test_img(nums[i-3])))
            threads[i].start() 
    for i in range(7):
        threads[i].join()

    final = ""
    for i in range(len(ans)):
        final += str(ans[i])
        if i == 2:
            final += '-'
    return(final)

def get_final():
    name = var2.get()
    ans = find_license(name)
    t = tk.Text(root, width=22, height=4, font=("Helvetica", 15), selectforeground='red')
    t.place(x=27, y=180)
    t.insert('insert',ans)

'''simple gui by tkinter'''
root = tk.Tk()
root.title('License plate recognition')
root.geometry('300x300')

line0 = tk.Label(root, bg = 'light cyan', text = "Image : ", font = ("Helvetica", 12)) 
line0.place(x=30, y=45)

var2 = tk.StringVar()
b2 = tk.Entry(root, textvariable = var2, font=("Helvetica", 12), show=None, width=20)
b2.place(x=85, y=45)

a = tk.Button(root, bg='light cyan', text="Get license!", width=25, height=2, command=get_final)
a.place(x=60, y=95)

root.mainloop()