import numpy as np 
import mnist
import matplotlib.pyplot as plt
import random

train_image,train_label=mnist.train_images(),mnist.train_labels()
test_images,test_labels=mnist.test_images(),mnist.test_labels()

train_image=train_image/255.0
test_images=test_images/255.0

train_image=train_image.reshape(-1,784)
test_images=test_images.reshape(-1,784)

inputlayer=784
hiddenlayer1=20
hiddenlayer2=20
outputlayer=10
learnrate=0.01


weights1=np.random.randn(inputlayer,hiddenlayer1)
bias1=np.zeros((1,hiddenlayer1))
weights2=np.random.randn(hiddenlayer1,hiddenlayer2)
bias2=np.zeros((1,hiddenlayer2))
weights3=np.random.randn(hiddenlayer2,outputlayer)
bias3=np.zeros((1,outputlayer))


def ReLU(x):
    return np.maximum(0,x)

def ReLUDer(x):
    return(x>0)

def Softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


classes=10
one_hot=np.zeros((len(train_label),classes))
for i,label in enumerate(train_label):
    one_hot[i,label]=1

epochs=100
batch=32
for epoch in range(epochs):
    totlos=0;
    for i in range(0,len(train_image),batch):
        batchimg=train_image[i:i+batch]
        batchlabel=one_hot[i:i+batch]
        h1in=np.dot(batchimg,weights1)+bias1
        h1out=ReLU(h1in)
        h2in=np.dot(h1out,weights2)+bias2
        h2out=ReLU(h2in)
        outputin=np.dot(h2out,weights3)+bias3
        outputout=Softmax(outputin)

        eps=1e-15
        loss=-np.sum(batchlabel*np.log(outputout+eps))/len(batchlabel)
        totlos+=loss

        delout=outputout-batchlabel
        grad_w3=np.dot(h2out.T,delout)
        grad_b3=np.sum(delout,axis=0,keepdims=True)

        delh2=np.dot(delout,weights3.T)*ReLUDer(h2in)
        grad_w2=np.dot(h1out.T,delh2)
        grad_b2=np.sum(delh2,axis=0,keepdims=True)

        delh1=np.dot(delh2,weights2.T)*ReLUDer(h1in)
        grad_w1=np.dot(batchimg.T,delh1)
        grad_b1=np.sum(delh1,axis=0,keepdims=True)

        weights1-=learnrate*grad_w1
        bias1-=learnrate*grad_b1
        weights2-=learnrate*grad_w2
        bias2-=learnrate*grad_b2
        weights3-=learnrate*grad_w3
        bias3-=learnrate*grad_b3
    print(f"Epoch{epoch+1}/{epochs},Loss:{totlos/len(train_image)}")

print("Training complete")

correct=0
tot=len(test_images)
for i in range(len(test_images)):
    testin=test_images[i]
    h1_in=np.dot(testin,weights1)+bias1
    h1_out=ReLU(h1_in)
    h2_in=np.dot(h1_out,weights2)+bias2
    h2_out=ReLU(h2_in)
    out_in=np.dot(h2_out,weights3)+bias3
    out_out=Softmax(out_in)
    pred=np.argmax(out_out)
    if pred==test_labels[i]:
        correct+=1
    accuracy=correct/tot
print(f"Test Acc:{accuracy*100}%")

samp=10
for _ in range(samp):
    i=np.random.randint(0,5999)
    testin=test_images[i]
    h1_in=np.dot(testin,weights1)+bias1
    h1_out=ReLU(h1_in)
    h2_in=np.dot(h1_out,weights2)+bias2
    h2_out=ReLU(h2_in)
    out_in=np.dot(h2_out,weights3)+bias3
    out_out=Softmax(out_in)
    pred=np.argmax(out_out)
    if pred==test_labels[i]:
        correct+=1
    accuracy=correct/tot
    plt.imshow(test_images[i].reshape(28,28),cmap='Greys')
    plt.title(f"Predicted:{pred}")
    plt.show()
