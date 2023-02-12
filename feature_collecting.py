from skimage import feature
import cv2
import numpy as np
import matplotlib.pyplot as plt

#calculate and returns the lbp histogram with "lbp_params" of the image "img"
def lbp_feature_correct(img, lbp_params):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ## Get the LBP Histogram
    lbp = feature.local_binary_pattern( gray, lbp_params["points"], lbp_params["radius"], lbp_params["method"] )
    n_bins = int(lbp.max() + 1)
    (hist, bins) = np.histogram(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins))
    return hist, lbp

## LBP parameters ##
lbp_params = dict(points=8,
    radius=2,
    method="nri_uniform")


def calculate_index_expression(img_pixel, index_name):

    Rb = float(img_pixel[0])
    Rg = float(img_pixel[1])
    Rr = float(img_pixel[2])
    R = Rr/255
    G = Rg/255
    B = Rb/255
    a = 0.667
    
    if index_name == "ngrdi":
        try:
            result = (Rg - Rr) / (Rg + Rr)
        except ZeroDivisionError:
            result = 0
    elif index_name == "gli":
        try:
            result = ((2*Rg) - Rr - Rb) / ((2*Rg) + Rr + Rb)
        except ZeroDivisionError:
            result = 0
    elif index_name == "rgbvi":
        try:
            result = ((Rg*Rg) - (Rr*Rb)) / ((Rg*Rg) + (Rr*Rb))
        except ZeroDivisionError:
            result = 0
    elif index_name == "ri":
        try:
            result = (Rr - Rg)/(Rr + Rg)
        except ZeroDivisionError:
            result = 0
    elif index_name == "cive":
        try:
            result = (0.441*Rr - 0.811*Rg + 0.385*Rb + 18.78745)/255
        except ZeroDivisionError:
            result = 0
    elif index_name == "mexg":
        try:
            result = (1.262*Rg - 0.884*Rr - 0.311*Rb)/255
        except ZeroDivisionError:
            result = 0
    elif index_name == "exg":
        try:
            r = R/(R+G+B)
        except ZeroDivisionError:
            r = 0
        try:
            g = G/(R+G+B)
        except ZeroDivisionError:
            g = 0
        try:
            b = G/(R+G+B)
        except ZeroDivisionError:
            b = 0
        
        result = 2*g-r-b
        return result
    elif index_name == "exgr":
        try:
            r = R/(R+G+B)
        except ZeroDivisionError:
            r = 0
        try:
            g = G/(R+G+B)
        except ZeroDivisionError:
            g = 0
        try:
            b = G/(R+G+B)
        except ZeroDivisionError:
            b = 0

        result = ((2*g-r-b) - (1.6*Rr - Rg))/255
        return result
    elif index_name == "vi":
        try:
            result = (Rg/((Rr**a) + (Rb**(1-a))))/255
        except ZeroDivisionError:
            result = 0
    return result

def calculate_vegetaion_index(img, index_name):
    data = img.shape

    #index histogram
    index_hist = np.zeros(10)
    
    #average histogram
    average_hist = np.zeros(2)
    
    average = 0
    average_green = 0
    count_pixels = 0
    
    for i in range(data[0]):
        for j in range(data[1]):
            index_value = calculate_index_expression(img[i,j], index_name)
            ##print(index_value)
            index_hist[int( (4.5 * index_value) + 4.5)] +=1
            
            #green component
            average_green+= float(img[i,i,1])            
            average+= ( (index_value+1) / 2)
            
            count_pixels +=1
            
            
    hist_sum = index_hist.sum()
    for i in range(len(index_hist)):
        index_hist[i] /= hist_sum

   
    #convert the media to a 0-1 scale
    average_hist[0] = (average/count_pixels) 
    average_hist[1] = (average_green/(count_pixels*255)) 
    
    #uncoment to not consider these averages
    average_hist = []

    #join ngrdi histogram with average histogram
    result_hist = np.concatenate((index_hist, average_hist), axis=None)        
    
    return result_hist

#return and calculate an histogram based on the H component of HSV representation of img
def calculate_hue_histogram(img):
    #Hue range is [0,179]

    hue_hist = np.zeros(10)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    data = img.shape
    for i in range(data[0]):
        for j in range(data[1]):
            hue_hist[int( (9/179) * hsv[i,j,0] )] +=1

    
    n_pixels = data[0] * data[1]
    #normalize the histogram
    for i in range(10):
        hue_hist[i] /= n_pixels
    
    return hue_hist


#return and calculate the final histogram (junction of lbp histogram and another histogram based on an index(RGVVI, GLI, VARI, NGRDI))
def calculate_final_histogram(img):
    
    #calculate lbp histogram
    lbp_hist , _ = lbp_feature_correct(img, lbp_params)
    #lbp_hist = []
    
    #calculate RGBVI histogram
    index_hist = calculate_vegetaion_index(img, "mexg")
    #index_hist = []
    
    #calculate Hue (HSV) histogram
    hue_hist = calculate_hue_histogram(img)
    #hue_hist = []

    #join the histograms
    final_hist = np.concatenate((lbp_hist, index_hist, hue_hist),axis=None)
    print(len(final_hist))
    return final_hist


    
f = open("features_lbp.txt", "w")
count_line = 0

f_dataset = open("annotations.txt", "r")
line = f_dataset.readline()

while(line != ''):
    array = line.split(' ')
    imageName = array[0]
    print("Nome da imagem: " + imageName)
    img = cv2.imread("../imagesOri/" + imageName + ".jpg")
    
    #resize
    imgResized = cv2.resize(img, (640,480))
    
    cv2.imwrite("Dataset/"+ imageName + ".jpg", imgResized)
    #get 3 rectangles from the image
    #cv2.rectangle(imgResized, (227,10), (414, 150), (0,0,255), 2)
    rectangle1 = imgResized[10:150, 227:414]
    rectangle2 = imgResized[160:300, 227:414]
    rectangle3 = imgResized[310:450, 227:414]
    
    final_hist = calculate_final_histogram(rectangle1)
    #save the final histogram
    np.savetxt(f, final_hist, newline=' ')
    f.write("\n")
    count_line+=1
    
    final_hist = calculate_final_histogram(rectangle2)
    #save the final histogram
    np.savetxt(f, final_hist, newline=' ')
    f.write("\n")
    count_line+=1

    final_hist = calculate_final_histogram(rectangle3)
    #save the final histogram
    np.savetxt(f, final_hist, newline=' ')
    f.write("\n")
    count_line+=1

    cv2.destroyAllWindows()
    
    line = f_dataset.readline()

print("Linhas guardadas "+ str(count_line))