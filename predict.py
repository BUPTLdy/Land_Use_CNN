# coding: utf-8

import uuid
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体,解决中文显示问题
# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray' 
label_size = 18
plt.rcParams['xtick.labelsize'] = label_size 
plt.rcParams['ytick.labelsize'] = label_size  # use grayscale output rather than a (potentially misleading) color heatmap
NUM_STYLE_LABELS=21


import sys
caffe_root='/home/ldy/workspace/caffe/' #设置你caffe的安装目录
sys.path.insert(0,caffe_root+'python')
import caffe                            #导入caffe
import time



caffe.set_mode_cpu()

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



style_label_file = caffe_root + 'examples/finetune_UCMerced_LandUse/style_names.txt'
style_labels = list(np.loadtxt(style_label_file, str, delimiter='\n'))
#print style_labels

#print style_labels

def disp_preds(net, image, labels, k=5, name='ImageNet'):
    net.blobs['data'].data[0, ...] = image
    probs = net.forward(start='conv1')['probs'][0]
    #print probs
    top_k = (-probs).argsort()[:k]
    probs_k=[]
    lables_k=[]
    for i, p in enumerate(top_k):
        
        probs_k.append(100*probs[p])
        lables_k.append(labels[p])


#    print 'top %d predicted %s labels =' % (k, name)
#    print '\n'.join('\t(%d) %5.2f%% %s' % (i+1, 100*probs[p], labels[p])
#                    for i, p in enumerate(top_k))
    return probs_k,lables_k

def disp_style_preds(net, image):
    probs_k,lables_k=disp_preds(net, image, style_labels, name='UCMerced_LandUse')
    return probs_k,lables_k
    



def show_labes(image,probs,lables,true_label):
    fig = plt.figure()
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
        ax1.text(v + 1, i, '%5.2f%%' %v,fontsize=14)

    plt.title('Probability Output',fontsize=20)
    ax2 = plt.subplot(gs[1])
    ax2.axis('off')
    ax2.imshow(image)
    plt.title(true_label,fontsize=20)
    plt.show()
    if true_label !=lables[0]:
        unique_filename = uuid.uuid4()
        fig.savefig('predit_worng/'+str(unique_filename)+'.jpg')

def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    #print  data.shape
    
    plt.imshow(data)
    plt.axis('off')
    plt.show()

#print net.params
#print net.blobs

def vis_show():
    image='/home/ldy/workspace/caffe/data/UCMerced_LandUse/Images/airplane/airplane80.tif'
    
    image = caffe.io.load_image(image)
    transformed_image = transformer.preprocess('data', image)
    disp_style_preds(net, transformed_image)
    print "第一层卷积层滤波器可视化："
    filters = net.params['conv1'][0].data
    vis_square(filters.transpose(0, 2, 3, 1)) 
    
    print '原图像：'
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    print "第一层卷积层输出："
    feat = net.blobs['conv1'].data[0, :36]
    vis_square(feat)
    print "第二层卷积层输出："
    feat = net.blobs['conv2'].data[0, :36]
    vis_square(feat)
    print "第三层卷积层输出："
    feat = net.blobs['conv3'].data[0, :36]
    vis_square(feat)
    print "第四层卷积层输出："
    feat = net.blobs['conv4'].data[0, :36]
    vis_square(feat)
    print "第五层卷积层池化后输出："
    feat = net.blobs['pool5'].data[0, :36]
    vis_square(feat)
    
    #filters = net.params['conv2'][0].data
    #vis_square(filters[:48].reshape(48**2, 5, 5))
def show_predict():    
    images='/home/ldy/workspace/caffe/data/UCMerced_LandUse/test.txt'
    images = list(np.loadtxt(images, str, delimiter='\n'))
    for image in images:
        true_label=image.split('/')[-2]
        image = caffe.io.load_image(image.split(' ')[-2])
        t0=time.time()
        transformed_image = transformer.preprocess('data', image)
        probs,lables=disp_style_preds(net, transformed_image)
        t1=time.time()
        show_labes(image,probs,lables,true_label)
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
        probs,lables=disp_style_preds(net, transformed_image)
        if true_label==lables[0]:
            precalss_corrct_num[true_label]+=1
        #print i,true_label,preclass_num[true_label],precalss_corrct_num[true_label]
    preclass_acc={}
    for label in style_labels:
        preclass_acc[label]=float(precalss_corrct_num[label])/float(preclass_num[label])
    print preclass_acc
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
    plt.savefig('acc1.eps')
    plt.show()


#show_acc_preclass()        
#show_predict()
vis_show()