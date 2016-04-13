import sys
from time import time
import pandas as pd
from pandas import DataFrame 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools

#from sklearn import metrics
#from sklearn.cluster import KMeans
#from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from sklearn.cross_validation import StratifiedKFold
from sklearn.externals.six.moves import xrange
from sklearn.mixture import GMM


fileID = sys.argv[1];           #fileID
set = sys.argv[2];              #Set
numSpeakers = sys.argv[3];      #Number of Speakers
blockLength = sys.argv[4];      #Block length
hopLength = sys.argv[5];        #Hop length


path = "/Users/avrosh/Documents/Coursework/7100_Spring_16/Dataset/dataset/"+set+"/features/set"+set+"_"+hopLength+"_"+blockLength+"_S"+numSpeakers+"_"+fileID+".csv"

f = open(path)
f.readline()

data = np.loadtxt(fname = f, delimiter=',')

labels = data[:,0]
#print labels

#normalize data
features = scale(data[:,1:])
#features = data[:,1:]
#print features

# Break up the dataset into non-overlapping training (75%) and testing
# (25%) sets.
skf = StratifiedKFold(labels, n_folds=4)
# Only take the first fold.
train_index, test_index = next(iter(skf))


X_train = features[train_index]
y_train = labels[train_index]
X_test = features[test_index]
y_test = labels[test_index]

n_classes = len(np.unique(y_train))

n_samples, n_features = features.shape
n_speakers = len(np.unique(labels))
speaker_ids = np.unique(labels)
print speaker_ids
print ("n_speakers %d \nn_samples %d \nn_features %d" % (n_speakers,n_samples,n_features))

print(79 * '_')

def make_ellipses(gmm, ax):
    for n, color in enumerate('rgb'):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)


# Try GMMs using different types of covariances.
classifiers = dict((covar_type, GMM(n_components=n_classes,
                                    covariance_type=covar_type, init_params='wc', n_iter=20))
                   for covar_type in ['spherical', 'diag', 'tied', 'full'])

n_classifiers = len(classifiers)



########################################################################
#Visualize data

plt.figure(figsize=(3 * n_classifiers / 2, 6))
plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
                    left=.01, right=.99)

for index, (name, classifier) in enumerate(classifiers.items()):
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    classifier.means_ = np.array([X_train[y_train == i].mean(axis=0)
                                  for i in xrange(n_classes)])

#    print classifier.means_

    print(79 * '_')
        
    # Train the other parameters using the EM algorithm.
    classifier.fit(X_train)
  
    h = plt.subplot(2, n_classifiers / 2, index + 1)
    make_ellipses(classifier, h)
  
    for n, color in enumerate('rgb'):
        data = features[labels == n]
        plt.scatter(data[:, 0], data[:, 1], 0.8, color=color,
                      label='test')

#                                    for n, color in enumerate('rgb'):
#                                        data = features[labels == n]
#                                            plt.scatter(data[:, 0], data[:, 1], 0.8, color=color,
#                                                        label=iris.target_names[n])


                                      # Plot the test data with crosses
#for n, color in enumerate('rgb'):
#    data = X_test[y_test == n]
#    plt.plot(data[:, 0], data[:, 1], 'x', color=color)
#    
#    y_train_pred = classifier.predict(X_train)
#    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
#    plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
#             transform=h.transAxes)
#             
#    y_test_pred = classifier.predict(X_test)
#    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
#    plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
#          transform=h.transAxes)
#
#    plt.xticks(())
#    plt.yticks(())
#    plt.title(name)
#
plt.legend(loc='lower right', prop=dict(size=12))


plt.show()







