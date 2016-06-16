# -*- coding: utf-8 -*-
"""
Created on Mon May 16 18:54:06 2016

@author: ldy
"""

# coding: utf-8


import numpy as np

NUM_STYLE_LABELS=21


import sys
caffe_root='/home/ldy/workspace/caffe/' #设置你caffe的安装目录

image_root='/home/ldy/workspace/caffe/data/UCMerced_LandUse/Images/'
sys.path.insert(0,caffe_root+'python')
import caffe                            #导入caffe




caffe.set_mode_gpu()
print 'load the structure of the model...'
model_def = caffe_root + 'models/finetune_UCMerced_LandUse/deploy1.prototxt'
print 'load the weights of the model...'
model_weights = caffe_root + 'models/finetune_UCMerced_LandUse/weights_finally.pretrained.caffemodel'

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


def show_predict():    
    images='/home/ldy/workspace/caffe/data/UCMerced_LandUse/creat_lmdb.txt'
    images = list(np.loadtxt(images, str, delimiter='\n'))
    features=[]
    labels=[]
    for image in images:
        image_list=image.split(' ')
        true_label=image_list[-1]
        image = caffe.io.load_image(image_root+image_list[-2])
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[0, ...] = transformed_image
        net.forward(start='conv1')
        feat = net.blobs['fc7'].data.copy()
        label_num=int(true_label)
        features.append(feat)
        labels.append(label_num)

    return features,labels
    
features,labels=show_predict()
np.save('features.npy',features)
np.save('labels.npy',labels)