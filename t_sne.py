# -*- coding: utf-8 -*-
"""
Created on Tue May 31 09:49:36 2016

@author: ldy
"""
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
palette = np.array(sns.color_palette("hls", 21))
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from sklearn.utils import shuffle
caffe_root='/home/ldy/workspace/caffe/'
X=np.load('features.npy')
y=np.load('labels.npy')
#y=y.reshape((-1,1))
style_label_file = caffe_root + 'examples/finetune_UCMerced_LandUse/style_names.txt'

style_labels = list(np.loadtxt(style_label_file, str, delimiter='\n'))
print style_labels
X=X.reshape((2100,4096))

X,y = shuffle(X, y, random_state=0)

#X=X[0:1000]
#y=y[0:1000]
print X.shape,y.shape
X_tsne = TSNE(n_components=2,early_exaggeration=10.0,random_state=20160530).fit_transform(X)
#X_tsne = PCA().fit_transform(X)

import itertools

print 'end'
#fig=plt.figure()
#ax=Axes3D(fig)
import matplotlib

markers =matplotlib.markers.MarkerStyle.filled_markers

markers=marker = itertools.cycle(markers)

f = plt.figure(figsize=(16,8))


ax = plt.subplot(aspect='equal')



for i in xrange(21):
    ax.scatter(X_tsne[y==i, 0], X_tsne[y==i, 1],marker=markers.next(),c=palette[i],label=style_labels[i])
plt.legend(loc=2,numpoints=1, ncol=2, fontsize=12, bbox_to_anchor=(1.05, 0.8))
ax.axis('off')
plt.savefig('t_sne.eps')
plt.show()