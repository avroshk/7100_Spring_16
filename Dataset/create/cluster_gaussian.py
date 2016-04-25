import sys
from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
from scipy import linalg

from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from sklearn import mixture

###Get command line arguments
fileID = sys.argv[1];           #fileID
set = sys.argv[2];              #Set
numSpeakers = sys.argv[3];      #Number of Speakers
blockLength = sys.argv[4];      #Block length
hopLength = sys.argv[5];        #Hop length
thresholdOrder = sys.argv[6]    #Adaptive Threshold order
extraid = sys.argv[7];          #extraid


###Prepare output file path
outputRoot = "/Users/avrosh/Documents/Coursework/7100_Spring_16/Dataset/dataset/"+set+"/"+"set"+set+"_S"+numSpeakers+"_"+blockLength+"_"+fileID+"_"+thresholdOrder
if extraid != 0:
    outputRoot = outputRoot + "_" + extraid
outputRoot = outputRoot + "_gmm.csv"

txtResultFile = open(outputRoot, "w")


###Prepare inut file path
path = "/Users/avrosh/Documents/Coursework/7100_Spring_16/Dataset/dataset/"+set+"/features/set"+set+"_"+hopLength+"_"+blockLength+"_S"+numSpeakers+"_"+fileID+"_"+thresholdOrder
if extraid != 0:
    path = path + "_" + extraid

path = path + ".csv"

f = open(path)
f.readline()


###Read data
data = np.loadtxt(fname = f, delimiter=',')

all_labels = data[:,0]
labels = all_labels[all_labels != 0]
#labels = data[:,0]
#print labels

#normalize data
features = data[data[:,0] != 0]
features = scale(features[:,1:])
#features = scale(data[:,1:])
#features = data[:,1:]
#print features

n_classes = len(np.unique(labels))

print n_classes

n_samples, n_features = features.shape
n_speakers = len(np.unique(labels))
speaker_ids = np.unique(labels)
print speaker_ids
print ("n_speakers %d \nn_samples %d \nn_features %d" % (n_speakers,n_samples,n_features))

print(79 * '_')

sample_size = 300;

def gmm_clustering(estimator,name,data):
    t0 = time()
    
    estimator.fit(data)
    
    estimated_labels = estimator.predict(data)
    
    homogeneity_score = metrics.homogeneity_score(labels,estimated_labels)
    completeness_score = metrics.completeness_score(labels, estimated_labels)
    v_measure_score = metrics.v_measure_score(labels, estimated_labels)
    adjusted_rand_score = metrics.adjusted_rand_score(labels, estimated_labels)
    adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(labels, estimated_labels)
    silhouette_score = metrics.silhouette_score(features,  estimated_labels,
                                                metric='euclidean',
                                                sample_size=sample_size)
        
    
    
    i=0
    j=0
    for label in all_labels:
        i = i + 1;
        txtResultFile.write("{0}".format(label))
        txtResultFile.write(",")
        if label == 0:
            txtResultFile.write("{0}".format(-1))
        else:
            txtResultFile.write("{0}".format(estimated_labels[j]))
            j = j + 1
        if i<len(all_labels):
            txtResultFile.write("\n")



    print('Name: % 9s \n'
          'Time: %.2fs \n'
          'Homogeneity Score: %.3f \n'
          'Completeness Score: %.3f \n'
          'V Measure score: %.3f \n'
          'Adjusted rand score: %.3f \n'
          'Adjusted Mutual Info score: %.3f \n'
          'Silhouette Score: %.3f'
          % (name, (time()-t0),
             homogeneity_score,
             completeness_score,
             v_measure_score,
             adjusted_rand_score,
             adjusted_mutual_info_score,
             silhouette_score))

##GMM
# Fit a mixture of Gaussians with EM using five components
gmm = mixture.GMM(n_components=n_classes, covariance_type='full')
gmm_clustering(gmm,
              name='gmm',
              data=features)

##Visualize data

reduced_data = PCA(n_components=2).fit_transform(features)
gmm.fit(reduced_data)

color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm','k'])

for speaker in speaker_ids:
    
    speaker_labels = np.argwhere(labels==speaker)
    plt.scatter(reduced_data[speaker_labels,0],
                reduced_data[speaker_labels,1],
                color=next(color_iter))

for i, (clf, title) in enumerate([(gmm, 'GMM')]):
    splot = plt.subplot(1, 1, 1 + i)

    Y_ = clf.predict(reduced_data)

    for i, (mean, covar, color) in enumerate(zip(clf.means_, clf._get_covars(), color_iter)):
        
        v, w = linalg.eigh(covar)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
#        if not np.any(Y_ == i):
#            continue
#        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

#        print X[Y_ == i, 0]

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        
        splot.add_artist(ell)

    plt.xlim(-10, 10)
    plt.ylim(-6, 6)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

#g = mixture.GMM(n_components=n_classes)
#g.fit(features)
#print g
#GMM(covariance_type='diag', init_params='wmc', min_covar=0.001,
#    n_components=n_classes, n_init=1, n_iter=100, params='wmc',
#    random_state=None, thresh=None, tol=0.001, verbose=0)
#
#g_weights = np.round(g.weights_, n_classes)
#print g_weights
#g_means = np.round(g.means_, n_classes)
##print g_means
#g_covars = np.round(g.covars_, n_classes)
##print g_covars
#p = g.predict(features)
#print p
#
#reduced_data = PCA(n_components=2).fit_transform(features)
#
#g_PCA = mixture.GMM(n_components=n_classes)
#g_PCA.fit(reduced_data)
#h = .02




## Try GMMs using different types of covariances.
#classifiers = dict((covar_type, GMM(n_components=n_classes, covariance_type=covar_type, init_params='wmc', n_iter=20)) for covar_type in ['spherical', 'diag', 'tied', 'full'])
#
##thegmm = GMM(n_components=n_classes,covariance_type='tied',init_params='wc',n_iter=20)
##thegmm.fit(features)
#
#n_classifiers = len(classifiers)



########################################################################
#Visualize data
#
#plt.figure(figsize=(3 * n_classifiers / 2, 6))
#plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
#                    left=.01, right=.99)
##
#for index, (name, classifier) in enumerate(classifiers.items()):
##    # Since we have class labels for the training data, we can
##    # initialize the GMM parameters in a supervised manner.
#    classifier.means_ = np.array([X_train[y_train == speaker_id].mean(axis=0) for speaker_id in speaker_ids])
##
##    print classifier.means_
###
##    print index
#    print(79 * '_')
##
##    # Train the other parameters using the EM algorithm.
#    classifier.fit(X_train)
##    a = classifier.predict(features)
#
##    print a
##
#    h = plt.subplot(2, n_classifiers / 2, index + 1 )
##    make_ellipses(classifier, h)
##
#    for n, color in enumerate('rgbk'):
#        speaker_id = speaker_ids[n]
##        print speaker_id
#        data = features[labels == speaker_id]
##        reduced_data = PCA(n_components=2).fit_transform(data)
#        print data
#        plt.scatter(data[:, 0], data[:, 1], 0.8, color=color, label=speaker_id)
##
##    for n, color in enumerate('rgb'):
##        data = features[labels == n]
##            plt.scatter(data[:, 0], data[:, 1], 0.8, color=color,
##                        label=iris.target_names[n])
#
#
#    # Plot the test data with crosses
##    for n, color in enumerate('rgbk'):
##        speaker_id = speaker_ids[n]
##        data = X_test[y_test == speaker_id]
##        plt.plot(data[:, 0], data[:, 1], 'x', color=color)
###
##    y_train_pred = classifier.predict(X_train)
###    print y_train_pred
##    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
##    plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
##             transform=h.transAxes)
###
##    y_test_pred = classifier.predict(X_test)
###    print y_test_pred
##    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
##    plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
##          transform=h.transAxes)
##
#    plt.xticks(())
#    plt.yticks(())
#    plt.title(name)
##
#plt.legend(loc='lower right', prop=dict(size=12))
#
#
plt.show()






