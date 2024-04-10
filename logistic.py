from data import *
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.special import expit
import warnings



warnings.filterwarnings("ignore") #agnohsh twn errror poy prokyptoun sto idle

data=Data()

L=[4] #λ



n=0
Negative_Train=[]
Positive_Train=[]
dianisma=[] #train dedomena
Negative_Test=[]
Positive_Test=[]
weights=[]
Final_weights=[]
dev_data=[]
temp=[]

#"""----------------------Train function--------------------------------"""

def train(train_files): #train_files=pososto twn review poy 8eloyme na e3etasoyme
    global Negative_Train,Positive_Train,n,dianisma,weights,dev_data,temp
    if len(temp)==0:
        print('LOading training data...')
        Negative_Train.extend(data.get01('train','neg'))
        Positive_Train.extend(data.get01('train','pos'))
        temp=Negative_Train+Positive_Train
        random.shuffle(temp)

    dev_data=temp[int(len(temp)*train_files/100):]
    dianisma=temp[:int(len(temp)*train_files/100)]
    n=len(dianisma)

    print('Training....')
    print('Calculating Weights...')
    weights=np.zeros((len(dianisma[0])-1))
    Stoxastiki_anabasi()
    print('Selecting best weights...')
    best_weights()

def Stoxastiki_anabasi(): #Synarthsh stoxastiki anabashs klhshs.
    global weights,dianisma,L,Final_weights
    for lv in L:
        Counter=0
        Max=4
        cost=[-1,0]
        a=0.01 #learing_rate

        for i in range(len(weights)):
            weights[i]=random.uniform(-3.14,3.14)
        while Counter < Max and not syglisi(cost[0],cost[1]):
            Counter=Counter+1
            random.shuffle(dianisma)
            for i in range(n):
                cost[0]=cost[1]
                cost[1]=cost[1]+kanonikopoihsh(i,lv)
                predicted_value = sigmoid_function(dianisma[i],weights)
                y=dianisma[i][-1]
                weights[0]=weights[0]-a*(predicted_value -y)*dianisma[i][0]
                for k in range(1,len(weights)):
                    weights[k]=weights[k]-a*((predicted_value-y)*dianisma[i][k] - lv/n*weights[k])
        Final_weights.append(weights)

def syglisi(old_cost,new_cost):  #elegxoyme an syglinei to kostos
    if abs(new_cost-old_cost)<pow(10,-3):
        return True
    return False


def kanonikopoihsh(x,L):
    kanonik_value=0
    for i in range(1,len(weights)):
        kanonik_value=kanonik_value+np.square(weights[i])
    kanonik_value=kanonik_value*L/(2*n)
    h=sigmoid_function(dianisma[x],weights)
    return (-dianisma[x][-1]*np.log(h) -(1-dianisma[x][-1])*np.log(1-h)) + kanonik_value


def sigmoid_function(x,w): 
    return expit(np.dot(w, x[:-1]))#1/(1+np.exp(-np.dot(weights,x[:-1])))


def best_weights():  #epilogh kalyterwn varwn.
    global weights,Final_weights
    w_cost=[]
    for i in Final_weights:
        Counter_error=0
        for j in range(len(dev_data)):
            if sigmoid_function(dev_data[j],i)>=0.5 and dev_data[j][-1] == 0:
                Counter_error=Counter_error+1
            elif sigmoid_function(dev_data[j],i)<0.5 and dev_data[j][-1] == 1:
                Counter_error=Counter_error+1
        w_cost.append(Counter_error)
    weights=Final_weights[w_cost.index(min(w_cost))]
    

#"""--------------------------------------------test-------------------------------------------"""

def test():
    global NegativeTest,Positive_Test,dianisma
    correct_pos=0
    wrong_pos=0
    correct_neg=0
    wrong_neg=0
    dianisma=[]
    print('Loading testing data....')
    Positive_Test=data.get01('test','pos')
    Negative_Test=data.get01('test','neg')
    dianisma=Positive_Test+Negative_Test
    random.shuffle(dianisma)
    for i in range(len(dianisma)):
        if sigmoid_function(dianisma[i],weights)>=0.50:
            if dianisma[i][-1] ==1:
                correct_pos=correct_pos +1
            else:
                wrong_pos=wrong_pos+1
        else:
            if dianisma[i][-1]==1:
                wrong_neg=wrong_neg+1
            else:
                correct_neg=correct_neg+1
            
                
    Positive_precision = correct_pos/(correct_pos+wrong_pos)
    Negative_precision = correct_neg/(correct_neg+wrong_neg)
    precision=0.5*(Positive_precision+Negative_precision)

    
    Positive_recall= correct_pos/(correct_pos+wrong_neg)
    Negative_recall=correct_neg/(correct_neg+wrong_pos)
    recall=0.5*(Positive_recall+Negative_recall)

    print("Accuracy: ",(correct_neg+correct_pos)/len(dianisma)*100,"%")
    print("Precision: ",precision*100,"%")
    print("Recall: ",recall*100,"%")

#"""------------------------------Graph------------------------------------------"""
def graph1():
    randomV=[10,20,30,40,50,60,70,80,90] #random train files(%) 
    accuracy1=[]
    accuracy2=[]
    for v in randomV:
        
        correct_pos=0
        correct_neg=0
        
        wrong_pos=0
        wrong_neg=0
        
        train(v)
        for i in range(n):
            if sigmoid_function(dianisma[i],weights)>=0.50:
                if dianisma[i][-1]==0:
                    wrong_pos = wrong_pos+1
                else:
                    correct_pos = correct_pos+1
            else:
                if dianisma[i][-1]==1:
                    wrong_neg=wrong_neg+1
                else:
                    correct_neg=correct_neg+1
        accuracy1.append((correct_neg+correct_pos)/len(dianisma))

        correct_pos=0
        correct_neg=0
        
        wrong_pos=0
        wrong_neg=0
        for i in range(len(dev_data)):
            if sigmoid_function(dev_data[i],weights)>=0.5:
                if dev_data[i][-1]==1:
                    correct_pos = correct_pos+1
                else:
                    wrong_pos = wrong_pos+1
            else:
                if dev_data[i][-1]==1:
                    wrong_neg=wrong_neg+1
                else:
                    correct_neg=correct_neg+1
        accuracy2.append((correct_neg+correct_pos)/len(dev_data))
    plt.plot(accuracy1,label='train data')
    plt.plot(accuracy2,label='dev data')
    plt.title("Accuracy")
    plt.xlabel("Examples")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
                    
       
def graph2():
    
    orio=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    precision=[]
    recall=[]
    correct_pos=0
    correct_neg=0
        
    wrong_pos=0
    wrong_neg=0
    train(80)
    for o in orio:
        for i in range(len(dev_data)):
            if sigmoid_function(dev_data[i],weights)>=o:
                if dev_data[i][-1]==0:
                    wrong_pos = wrong_pos+1
                else:
                    correct_pos = correct_pos+1
            else:
                if dev_data[i][-1]==1:
                    wrong_neg=wrong_neg+1
                else:
                    correct_neg=correct_neg+1
                    
        Positive_precision=correct_pos/(correct_pos+wrong_pos)
        Negative_precision=correct_neg/(correct_neg+wrong_neg)

        Positive_recall = correct_pos/(correct_pos+wrong_neg)
        Negative_recall = correct_neg/(correct_neg+wrong_pos)

        precision.append(0.5*(Positive_precision + Negative_precision))
        recall.append(0.5*(Positive_recall+Negative_recall))

    plt.plot(recall,label='recall')
    plt.plot(precision,label='precision')
    plt.title("Precision/Recall")
    plt.xlabel("Example")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


#"""---------------------------Programm_running----------------------------------""" 
def programm_running():
    print("Helloo!")
    train(100)
    print("Training is complete!\nContinue to testing...")
    test()
    print("Testing is complete!\nProgramm is closing...\nThank you!")
    #graph1()
    #graph2()
    
programm_running()
