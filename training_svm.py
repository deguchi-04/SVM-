import cv2
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import sklearn

#function to append the values to each class
def label_to_percentage(label_list):
    percentage_list = []
    for i in label_list:
        if i == 0:
            percentage_list.append(0)
        elif i == 1:
            percentage_list.append(33)
        elif i == 2:
            percentage_list.append(66)
        elif i == 3:
            percentage_list.append(100)
    return percentage_list

def plot_confusion_matrix(labels, pred_labels):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    cm = confusion_matrix(labels, pred_labels,  sample_weight=None, normalize='true');
    cm = ConfusionMatrixDisplay(cm, display_labels =classes);
    cm.plot(values_format='f', cmap='Blues', ax=ax)

classes = ('0', '33', '66', '100')


def label_to_percentage(label_list):
    percentage_list = []
    for i in label_list:
        if i == 0:
            percentage_list.append(0)
        elif i == 1:
            percentage_list.append(33)
        elif i == 2:
            percentage_list.append(66)
        elif i == 3:
            percentage_list.append(100)
    return percentage_list

feature_list = []
label_list = []
image_name_list = []
line_aux_list = []
f_features = open("features_lbp_gli_hue_10_05.txt", "r")

#read features file
line = f_features.readline()
while line != '':

    array = line.split(' ')
    array.pop()
    feature_list.append(array)
    line = f_features.readline()

#read dataset file
f_dataset = open("demofile_10_05.txt", "r")

line = f_dataset.readline()
while line != '':
    line_aux_list.append(line)
    array = line.split(' ')
    image_name_list.append(array[0])
    array.pop(0)
    array.pop()
    for i in array:
        if i == '0':
            i = 0
        elif i == '33':
            i = 1
        elif i == '66':
            i = 2
        elif i == '100':
            i = 3
        else:
            print(i)
            print("ERROR:Invalid value")

        label_list.append(i)

    line = f_dataset.readline()

#separate in training and testing data


data_train = feature_list[285:len(feature_list)]
data_label_train = label_list[285:len(feature_list)]

data_test = feature_list[:285]
data_label_test = label_list[:285]



print(len(data_train))
print(len(data_test))

clf = make_pipeline(StandardScaler(), SVC(C=1,gamma='auto'))
clf.fit(data_train, data_label_train)

#print(data_test[0])
#print(clf.predict(data_test.reshape(1,)))
results = clf.predict(data_test)
score = f1_score(data_label_test, results, average='macro') #an arithmetic mean of the per-class F1-scores. We gave equal weights to each class
print("\nF1 Score Macro of the worst descriptor: ", score)

score = f1_score(data_label_test, results, average='micro') #look at all the samples together, equal to accuracy
print("\nF1 Score Micro of the worst descriptor: ", score)

score = f1_score(data_label_test, results, average='weighted')  #we weight the F1-score of each class by the number of samples from that class
print("\nF1 Score Weighted of the worst descriptor: ", score)

score = f1_score(data_label_test, results, average=None)
print("\nLabels score of the worst descriptor: ", score)

acc_score = accuracy_score(data_label_test, results)
print("\nAccuracy score of the worst descriptor: ", acc_score)


list_range = 285/3
data_index = 0
i=list_range # colocar aqui um 0 para mostar imagens

x = 0
results_percentage = label_to_percentage(results)
data_label_test_percentage = label_to_percentage(data_label_test)

#print(results)
#print(len(results))

#print(data_label_test)

#print(image_name_list)

while i < list_range:
    #print(image_name_list[i])
    
    print(line_aux_list[i])
    imgOrigin = cv2.imread("aveleda_2020_07_23_zed_images/" + image_name_list[i] + ".jpg")
    imgResized = cv2.resize(imgOrigin, (640,480))
    cv2.rectangle(imgResized, (227,10), (414, 150), (0,0,255), 2)
    cv2.rectangle(imgResized, (227,160), (414, 300), (0,0,255), 2)
    cv2.rectangle(imgResized, (227,310), (414, 450), (0,0,255), 2)
    """
    imgResized = cv2.putText(imgResized, "Pred", (115, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,255), 2, cv2.LINE_AA)
    imgResized = cv2.putText(imgResized, "True", (440, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,255), 2, cv2.LINE_AA)
    imgResized = cv2.putText(imgResized, str(results_percentage[data_index]), (105, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0,255), 2, cv2.LINE_AA)
    imgResized = cv2.putText(imgResized, str(results_percentage[data_index+1]), (105, 260), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
    imgResized = cv2.putText(imgResized, str(results_percentage[data_index+2]), (105, 410), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
    imgResized = cv2.putText(imgResized, str(data_label_test_percentage[data_index]), (430, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
    imgResized = cv2.putText(imgResized, str(data_label_test_percentage[data_index+1]), (430, 260), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
    imgResized = cv2.putText(imgResized, str(data_label_test_percentage[data_index+2]), (430, 410), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
    """

    data_index+=3
    cv2.imshow("Leaf Detection", imgResized)
    cv2.waitKey(500)
    #print("data_index", data_index)
    if x == 0:
        x = 0
        read = input("Left - pred_value Right - true_value?: ")
    i+=1

matrix = confusion_matrix(data_label_test, results, sample_weight=None, normalize='true')
diagonal_area = 0
for i in range(4):
    diagonal_area += matrix[i][i]


    
plot_confusion_matrix(data_label_test, results)
print("\nDiagonal area:", (diagonal_area/4))
