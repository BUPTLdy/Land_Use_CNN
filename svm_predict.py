# -*- coding: utf-8 -*-
"""
Created on Mon May 16 18:54:06 2016

@author: ldy
"""

# coding: utf-8


import numpy as np
from sklearn.externals import joblib
NUM_STYLE_LABELS=21
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sys
caffe_root='/home/ldy/workspace/caffe/' #设置你caffe的安装目录

image_root='/home/ldy/workspace/caffe/data/UCMerced_LandUse/Images/'
sys.path.insert(0,caffe_root+'python')
import caffe                            #导入caffe
import time




caffe.set_mode_gpu()
print 'load the structure of the model...'
model_def = caffe_root + 'models/finetune_UCMerced_LandUse/deploy1.prototxt'
print 'load the weights of the model...'
model_weights = caffe_root + 'models/finetune_UCMerced_LandUse/weights.pretrained.caffemodel'

print 'build the trained net...'
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load('/home/ldy/workspace/caffe/examples/finetune_UCMerced_LandUse/mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
#print 'mean-subtracted values:', zip('BGR', mu),mu

# create trasformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR


def show_labes(image,probs,lables,true_label):
    gs = gridspec.GridSpec(1, 2,width_ratios=[1,1],height_ratios=[1,1])
    ax1 = plt.subplot(gs[0])
    x = list(reversed(lables))
    y = list(reversed(probs))  
    colors=['#edf8fb','#b2e2e2','#66c2a4','#2ca25f','#006d2c']
    #colors = ['#624ea7', 'g', 'yellow', 'k', 'maroon']
    #colors=list(reversed(colors))
    width = 0.4 # the width of the bars 
    ind = np.arange(len(y))  # the x locations for the groups
    ax1.barh(ind, y, width, align='center', color=colors)
    ax1.set_yticks(ind+width/2)
    ax1.set_yticklabels(x, minor=False)
    for i, v in enumerate(y):
        ax1.text(v, i, '%5.2f%%' %v,fontsize=14)
    plt.title('Probability Output',fontsize=20)
    ax2 = plt.subplot(gs[1])
    ax2.axis('off')
    ax2.imshow(image)
#    fig = plt.gcf()
#    fig.set_size_inches(8, 6)
    plt.title(true_label,fontsize=20)
    plt.show()
    
style_label_file = caffe_root + 'examples/finetune_UCMerced_LandUse/style_names.txt'
style_labels = list(np.loadtxt(style_label_file, str, delimiter='\n'))
if NUM_STYLE_LABELS > 0:
    style_labels = style_labels[:NUM_STYLE_LABELS]
    
print style_labels,type(style_labels)
clf = joblib.load('svm_model/svm.pkl')  

def return_maxkoflist(arr,k):
    arr=np.array(arr[0])
   # print arr.size,arr
    index=arr.argsort()[-k:][::-1]
   # print index.size
    proba=arr[index]
    proba=proba*100
    labels=np.array(style_labels)
    index=labels[index]
    #print index
    return proba,index



def disp_style_preds(feat):
    proba=clf.predict_proba(feat)
    probas,labels=return_maxkoflist(proba,5)
    return probas,labels
    

def show_predict():    
    images='/home/ldy/workspace/caffe/data/UCMerced_LandUse/test.txt'
    images = list(np.loadtxt(images, str, delimiter='\n'))
    for image in images:
        true_label=image.split('/')[-2]
        image = caffe.io.load_image(image.split(' ')[-2])
        t0=time.time()
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[0, ...] = transformed_image
        net.forward(start='conv1')
        feat = net.blobs['fc7'].data.copy()
        probas,labels=disp_style_preds(feat)
        t1=time.time()
        show_labes(image,probas,labels,true_label)
        print '每张图片预测时间：%.3f s'%(t1-t0)

def show_acc_preclass():
    images='/home/ldy/workspace/caffe/data/UCMerced_LandUse/test.txt'
    images = list(np.loadtxt(images, str, delimiter='\n'))
    preclass_num={}
    precalss_corrct_num={}
    for label in style_labels:
        preclass_num[label]=0
        precalss_corrct_num[label]=0
        
    
    for i,image in enumerate(images):
        
        true_label=image.split('/')[-2]
        preclass_num[true_label]=preclass_num[true_label]+1
        image = caffe.io.load_image(image.split(' ')[-2])
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[0, ...] = transformed_image
        net.forward(start='conv1')
        feat = net.blobs['fc7'].data.copy()
        probas,lables=disp_style_preds(feat)
        if true_label==lables[0]:
            precalss_corrct_num[true_label]+=1
        #print i,true_label,preclass_num[true_label],precalss_corrct_num[true_label]
        
        
        
    preclass_acc={}
    for label in style_labels:
        preclass_acc[label]=float(precalss_corrct_num[label])/float(preclass_num[label])
    #print preclass_acc
    k=1
    plt.figure(figsize=(8,7))
    #plt.tight_layout()
    ind = np.arange(0, k*len(preclass_acc), k) 
    
    colors=['#edf8fb','#ccece6','#99d8c9','#66c2a4','#41ae76','#238b45','#005824','#f1eef6','#d4b9da','#c994c7','#df65b0','#e7298a','#ce1256','#91003f','#ffffb2','#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#b10026']
    rects=plt.bar(ind, preclass_acc.values(),width=0.7,color=colors)
    plt.xticks(ind+0.35, preclass_acc.keys(),rotation='vertical')
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%.2f' %float(height),
                ha='center', va='bottom')
    plt.xlim([0,ind.size])
    plt.tight_layout()
    plt.savefig('acc2.eps')
    plt.show()
    
#show_acc_preclass()   
show_predict()