import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random

##### read image #####
imagepath = 'input your image name'
image = cv2.imread(imagepath)

##### BGR2RGB #####
image = image[:,:,::-1]

##### RGB2B #####
def color_img(RGB_image):
    B_image = np.zeros((len(image),len(image[0]),3),np.uint8)
    for i in range(len(image)):
        for j in range(len(image[0])):
            B_image[i][j]= RGB_image[i][j]
            B_image[i][j][0] = 0
            B_image[i][j][1] = 0
    return(B_image)
        
##### RGB2GRAY #####
def gray_img(image):
    gray_image = np.zeros((len(image),len(image[0]),3),np.uint32)
    for i in range (len(image)):
        for j in range (len(image[0])):
            b,g,r = image[i][j]
            gray = (int(b)+int(g)+int(r))/3
            gray_image[i][j] = np.uint8(gray)
    return(gray_image)
        
##### move image #####
def location_img(RGB_image,moveAmount):
    image_location = np.zeros((len(image)+moveAmount,len(image[0])+moveAmount,3),np.uint32)
    for i in range(len(image)):
        for j in range(len(image[0])):
            image_location[i][j+moveAmount] = RGB_image[i][j]
    return(image_location)

##### zoom_amplify #####
def amplify_img(RGB_image,amplify_high,amplify_wide):
    amplify_image = np.zeros((len(image)*amplify_high,len(image[0])*amplify_wide,3),np.uint32)
    for i in range(len(image)):
        for j in range(len(image[0])):
            for k in range(amplify_high):
                amplify_image[amplify_high*i+k][amplify_wide*j:amplify_wide*(j+1)] = RGB_image[i][j]
    return(amplify_image)
        
##### zoom_shrink #####
def shrink_img(RGB_image,shrink_high,shrink_wide):
    shrink_image = np.zeros((len(RGB_image)//shrink_high,len(RGB_image[0])//shrink_wide,3),np.uint32)
    if shrink_high == 1 and shrink_wide == 1:
        for i in range(len(shrink_image)):
            for j in range(len(shrink_image[0])):
                shrink_image[i][j] = RGB_image[i][j]
    else:
        for i in range(len(shrink_image)):
            for j in range(len(shrink_image[0])):
                shrink_image[i][j] = RGB_image[shrink_high*i][shrink_wide*j]
    return(shrink_image)

##### rotation #####
def rot_img(shrink_image,angle):
    rot_image = np.zeros((len(RGB_image),len(RGB_image[0]),3),np.uint32)
    (h, w) = image.shape[:2]
    (h1, w1) = shrink_image.shape[:2]
    center = (h // 2, w // 2)
    center1 = (h1 // 2, w1 // 2)
    for i in range(len(shrink_image)):
        for j in range(len(shrink_image[0])):
            r_1 = int(math.cos(angle *math.pi)*(i-center1[0]) - math.sin(angle *math.pi)*(j - center1[1]) + center1[0])
            r_2 = int(math.cos(angle *math.pi)*(j-center1[1]) + math.sin(angle *math.pi)*(i - center1[0]) + center1[1])
            r_1 += (center[0]-center1[0])
            r_2 += (center[1]-center1[1])
            r_1 = np.uint32(np.clip(r_1,0,len(rot_image)-1))
            r_2 = np.uint32(np.clip(r_2,0,len(rot_image[0])-1))
            rot_image[r_1][r_2] = shrink_image[i][j]
    for i in range(len(rot_image)):
        for j in range(len(rot_image[0])):
            if rot_image[i][j][0] == 0 and rot_image[i][j][1] == 0 and rot_image[i][j][2] == 0:
                if j < len(rot_image[0])-1:
                    rot_image[i][j] = rot_image[i][j+1]
                else:
                    rot_image[i][j] = rot_image[i][j-1]
    return(rot_image)

##### gamma_transform #####
def gamma_trans(RGB_image,gamma):
    gamma_image = np.zeros((len(RGB_image),len(RGB_image[0]),3),np.uint32)
    gray_image = gray_img(RGB_image)
    for i in range(len(gray_image)):
        for j in range(len(gray_image[0])):
            gamma_value = int(255 * ((gray_image[i][j][0] / 255) ** gamma))
            gamma_value = np.uint32(np.clip(gamma_value,0,255))
            gamma_image[i][j] = np.uint32(gamma_value)
    return(gamma_image)

##### noise image #####
def noise_img(RGB_image):
    noise_image = np.zeros((len(RGB_image),len(RGB_image[0]),3),np.uint32)
    noise_image = noise_image + RGB_image
    rows, cols, chn = noise_image.shape
    for i in range(1500):
        x = np.random.randint(0, rows)
        y = np.random.randint(0, cols)
        noise_image[x,y,:] = 255
    return(noise_image)

##### sobel #####
def sobel(RGB_image):
    gray_image = gray_img(RGB_image)
    value_image = np.zeros((len(RGB_image),len(RGB_image[0]),3),np.uint32)
    value_max = 0
    for i in range (len(RGB_image)-2):
        for j in range (len(RGB_image[0])-2):
            value1 = (-1) * gray_image[0+i][0+j][0] + 1 * gray_image[0+i][0+j+2][0] + \
                    (-2) * gray_image[0+i+1][0+j][0] + 2 * gray_image[0+i+1][0+j+2][0] + \
                    (-1) * gray_image[0+i+2][0+j][0] + 1 * gray_image[0+i+2][0+j+2][0]
            value2 = (-1) * gray_image[0+i][0+j][0] + 1 * gray_image[0+i+2][0+j][0] + \
                    (-2) * gray_image[0+i][0+j+1][0] + 2 * gray_image[0+i+2][0+j+1][0] + \
                    (-1) * gray_image[0+i][0+j+2][0] + 1 * gray_image[0+i+2][0+j+2][0]
            value_mean = int(math.sqrt(value1**2 + value2**2))
            value_mean = np.uint32(np.clip(value_mean,0,255))
            value_image[i+1][j+1] = np.uint32(value_mean)
    return(value_image)

##### otsu #####
def otsu(RGB_image):
    gray_tower = gray_img(RGB_image)
    gray_tower.astype(np.float32)
    h = gray_tower.shape[0]
    w = gray_tower.shape[1]
    threshold = 0
    max_variance = 0
    for t in range(256):
        former = 0
        background = 0
        former_gray = 0
        background_gray = 0
        all_gray = 0
        for i in range(len(gray_tower)):
            for j in range(len(gray_tower[0])):
                if gray_tower[i][j][0] < t:
                    former_gray += gray_tower[i][j][0]
                    former += 1
                if gray_tower[i][j][0] >= t:
                    background_gray += gray_tower[i][j][0]
                    background += 1
                all_gray += gray_tower[i][j][0]
        if former and background != 0:
            former_rate = former / (h*w)
            background_rate = background / (h*w)
            all_gray_mean = all_gray / (h*w)
            former_gray_mean = former_gray / former
            background_gray_mean = background_gray / background
            variance = former_rate * ((former_gray_mean - all_gray_mean) ** 2) + \
                       background_rate * ((background_gray_mean - all_gray_mean) ** 2)
            if variance > max_variance:
                max_variance = variance
                threshold = t
    for i in range (len(gray_tower)):
        for j in range (len(gray_tower[0])):
            if gray_tower[i][j][0] < threshold:
                gray_tower[i][j] = np.uint32(0)
            if gray_tower[i][j][0] >= threshold:
                gray_tower[i][j] = np.uint32(255)
    return(gray_tower)

##### mean filtering #####
def mean_filter(noise_image):
    #noise_image = noise_img(RGB_image)
    denoise_image = np.zeros((len(noise_image),len(noise_image[0]),3),np.uint32)
    for i in range (len(noise_image)-2):
        for j in range (len(noise_image[0])-2):
            R_value = (1/16) * (int(noise_image[i][j][0]) + 2*int(noise_image[i][j+1][0]) + int(noise_image[i][j+2][0]) + \
                               2*int(noise_image[i+1][j][0]) + 4*int(noise_image[i+1][j+1][0]) + 2*int(noise_image[i+1][j+2][0]) + \
                               int(noise_image[i+2][j][0]) + 2*int(noise_image[i+2][j+1][0]) + int(noise_image[i+2][j+2][0]))
            G_value = (1/16) * (int(noise_image[i][j][1]) + 2*int(noise_image[i][j+1][1]) + int(noise_image[i][j+2][1]) + \
                               2*int(noise_image[i+1][j][1]) + 4*int(noise_image[i+1][j+1][1]) + 2*int(noise_image[i+1][j+2][1]) + \
                               int(noise_image[i+2][j][1]) + 2*int(noise_image[i+2][j+1][1]) + int(noise_image[i+2][j+2][1]))
            B_value = (1/16) * (int(noise_image[i][j][2]) + 2*int(noise_image[i][j+1][2]) + int(noise_image[i][j+2][2]) + \
                               2*int(noise_image[i+1][j][2]) + 4*int(noise_image[i+1][j+1][2]) + 2*int(noise_image[i+1][j+2][2]) + \
                               int(noise_image[i+2][j][2]) + 2*int(noise_image[i+2][j+1][2]) + int(noise_image[i+2][j+2][2]))
            denoise_image[i+1][j+1][0] = R_value
            denoise_image[i+1][j+1][1] = G_value
            denoise_image[i+1][j+1][2] = B_value
    return(denoise_image)

##### median filtering #####
def median_filter(image,i_kernal,j_kernal,times):
    median_image = np.zeros((len(image),len(image[0]),3),np.uint32)
    median_image += image
    for i in range (len(image)-(i_kernal-1)):
        for j in range (len(image[0])-(j_kernal-1)):
            R_value = []
            G_value = []
            B_value = []
            for k in range(i_kernal):
                for r in range (j_kernal):
                    R_value.append(image[i+k][j+r][0])
                    G_value.append(image[i+k][j+r][1])
                    B_value.append(image[i+k][j+r][2])
            R_med = np.median(R_value)
            G_med = np.median(G_value)
            B_med = np.median(B_value)
            median_image[i+(i_kernal//2)][j+(j_kernal//2)][0] = R_med
            median_image[i+(i_kernal//2)][j+(j_kernal//2)][1] = G_med
            median_image[i+(i_kernal//2)][j+(j_kernal//2)][2] = B_med
    times -= 1
    if times == 0:
        return(median_image)
    else:
        return(median_filter(median_image,i_kernal,j_kernal,times))

##### whilt2black #####
def whiteblack(imagepath):
    image = cv2.imread(imagepath,0)
    ret,a = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
    a_3 = np.zeros((len(a),len(a[0]),3),np.uint32)
    for i in range (len(a_3)):
        for j in range (len(a_3[0])):
            a_3[i][j] = 255 - a[i][j]
            #a_3[i][j] = a[i][j]
    return(a_3)

##### erode #####
def erode_img(bw_image,kernal_size,times):
    a_erode = np.zeros((len(bw_image),len(bw_image[0]),3),np.uint32)
    for i in range (len(bw_image)-(kernal_size-1)):
        for j in range (len(bw_image[0])-(kernal_size-1)):
            test = 0
            for k in range (kernal_size):
                for r in range (kernal_size):
                    if k == int(kernal_size/2) and r == int(kernal_size/2):
                        continue
                    if bw_image[i+k][j+r][0] == 255:
                        test += 1
            if int(test) == int(kernal_size*kernal_size-1):
                a_erode[i+int(kernal_size/2)][j+int(kernal_size/2)] = 255
            else:
                a_erode[i+int(kernal_size/2)][j+int(kernal_size/2)] = 0
    times -= 1
    if times == 0:
        return(a_erode)
    else:
        return(erode_img(a_erode,kernal_size,times))

##### dilate #####
def dilate_img(bw_image,kernal_size,times):
    a_dilate = np.zeros((len(bw_image),len(bw_image[0]),3),np.uint32)
    for i in range (len(bw_image)-(kernal_size-1)):
        for j in range (len(bw_image[0])-(kernal_size-1)):
            test = 0
            for k in range (kernal_size):
                for r in range (kernal_size):
                    if k == int(kernal_size/2) and r == int(kernal_size/2):
                        continue
                    if bw_image[i+k][j+r][0] == 255:
                        test += 1
            if int(test) >= 1:
                a_dilate[i+int(kernal_size/2)][j+int(kernal_size/2)] = 255
            else:
                a_dilate[i+int(kernal_size/2)][j+int(kernal_size/2)] = 0
    times -= 1
    if times == 0:
        return(a_dilate)
    else:
        return(dilate_img(a_dilate,kernal_size,times))
    
##### image_password #####
def split_img(love_img):
    love_img = whiteblack(imagepath5)
    #love_img = 255 - love_img
    image1 = np.zeros((2*len(love_img),2*len(love_img[0]),3),np.uint32)
    image2 = np.zeros((2*len(love_img),2*len(love_img[0]),3),np.uint32)

    square1 = np.array([[0,255],
                        [0,255]])
    square2 = np.array([[255,0],
                        [255,0]])
    square3 = np.array([[255,0],
                        [0,255]])
    square4 = np.array([[0,255],
                       [255,0]])
    square5 = np.array([[255,255],
                        [0,0]])
    square6 = np.array([[0,0],
                        [255,255]])

    squares1_black = (square1,square2)
    squares2_black = (square2,square1)
    squares3_black = (square3,square4)
    squares4_black = (square4,square3)
    squares5_black = (square5,square6)
    squares6_black = (square6,square5)

    squares1_white = (square1,square1)
    squares2_white = (square2,square2)
    squares3_white = (square3,square3)
    squares4_white = (square4,square4)
    squares5_white = (square5,square5)
    squares6_white = (square6,square6)

    all_black = [squares1_black,squares2_black,squares3_black,squares4_black,squares5_black,squares6_black]
    all_white = [squares1_white,squares2_white,squares3_white,squares4_white,squares5_white,squares6_white]

    for i in range (len(love_img)):
        for j in range (len(love_img[0])):
            k = random.randint(0,5)
            if love_img[i][j][0] == 0:
                image1[2*i:2*(i+1),2*j:2*(j+1),0] = all_black[k][0]
                image1[2*i:2*(i+1),2*j:2*(j+1),1] = all_black[k][0]
                image1[2*i:2*(i+1),2*j:2*(j+1),2] = all_black[k][0]
                image2[2*i:2*(i+1),2*j:2*(j+1),0] = all_black[k][1]
                image2[2*i:2*(i+1),2*j:2*(j+1),1] = all_black[k][1]
                image2[2*i:2*(i+1),2*j:2*(j+1),2] = all_black[k][1]
            if love_img[i][j][0] == 255:
                image1[2*i:2*(i+1),2*j:2*(j+1),0] = all_white[k][0]
                image1[2*i:2*(i+1),2*j:2*(j+1),1] = all_white[k][0]
                image1[2*i:2*(i+1),2*j:2*(j+1),2] = all_white[k][0]
                image2[2*i:2*(i+1),2*j:2*(j+1),0] = all_white[k][1]
                image2[2*i:2*(i+1),2*j:2*(j+1),1] = all_white[k][1]
                image2[2*i:2*(i+1),2*j:2*(j+1),2] = all_white[k][1]
    return(image1,image2)

##### image_password #####
def union_img(image1,image2):
    union_image = np.zeros((len(image1),len(image1[0]),3),np.uint32)
    for i in range (len(union_image)):
        for j in range (len(union_image[0])):
            value = image1[i][j][0] + image2[i][j][0]
            if value == 0:
                value_fix = 0
            elif value == 255:
                value_fix = 0
            else :
                value_fix = 255
            union_image[i][j] = value_fix
    return(union_image)

##### watermask #####
def img_watermask(image_goal,image_water,transparency):
    img_neg = np.zeros((len(image_water),len(image_water[0]),3),np.uint32)
    img_neg += image_water
    img_neg = 255 - img_neg
    for i in range (len(img_neg)):
        for j in range (len(img_neg[0])):
            img_neg[i][j][0] = int(img_neg[i][j][0] * transparency)
            img_neg[i][j][1] = int(img_neg[i][j][1] * transparency)
            img_neg[i][j][2] = int(img_neg[i][j][2] * transparency)
    img_neg = 255 - img_neg
    img_neg = shrink_img(img_neg,2,2)
    img_neg = img_neg / 255
    img_water = np.zeros((len(image_goal),len(image_goal[0]),3),np.uint32)
    img_water += image_goal
    h , w , d = img_neg.shape
    img_mask = img_water[-h:,-w:]
    for i in range(h):
        for j in range(w):
            img_mask[i][j][0] *= img_neg[i][j][0]
            img_mask[i][j][1] *= img_neg[i][j][1]
            img_mask[i][j][2] *= img_neg[i][j][2]
    img_water[-h:,-w:] = img_mask[-h:,-w:]
    return(img_water)