import sys
from time import time
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib as mpl
from scipy import linalg

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from sklearn import mixture

np.random.seed(42)

###Get command line arguments
clusterType = sys.argv[1]       #Clustering algorithm
fileIDMin = int(sys.argv[2]);           #fileID - min
fileIDMax = int(sys.argv[3]);           #fileID - max
set = sys.argv[4];              #Set
numSpeakers = sys.argv[5];      #Number of Speakers
blockLength = sys.argv[6];      #Block length
hopLength = sys.argv[7];        #Hop length
thresholdOrder = sys.argv[8]    #Adaptive Threshold order
extraid = int(sys.argv[9]);          #extraid

###Output file for aggregating results
resultsOutputFile = "/Users/avrosh/Documents/Coursework/7100_Spring_16/Dataset/dataset/"+set+"/"+"set"+set+"_S"+numSpeakers+"_"+blockLength+"_"+str(fileIDMin)+"-"+str(fileIDMax)+"_"+thresholdOrder
if extraid != 0:
    resultsOutputFile = resultsOutputFile + "_" + str(extraid)
resultsOutputFile = resultsOutputFile + "_" + clusterType + "_RESULT.csv"
# print outputRoot
    
txtFinalResultFile = open(resultsOutputFile, "w")

overall_completeness_score = 0
fileCount = 0


###Method
def visualize_gmm(data,gmm):
    ##Visualize data
    
    reduced_data = PCA(n_components=2).fit_transform(data)
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
        plt.show()


###Method
def visualize_kmeans(data):
    ########################################################################
    #Visualize data
    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++',n_clusters=n_speakers,n_init=10)
    kmeans.fit(reduced_data)
    
    #step size of mesh
    h = .02
    
    #Plot the decision boundary
    x_min, x_max = reduced_data[:,0].min() - 1, reduced_data[:,0].max() + 1
    y_min, y_max = reduced_data[:,1].min() - 1, reduced_data[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    #Obtain labels for each point in mesh
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    
    #Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')
        
    #Colour Cycler
    colorcycler = itertools.cycle(['r', 'g', 'b', 'y','c','k','w','m'])
            
            
    
    for speaker in speaker_ids:
        
        speaker_labels = np.argwhere(labels==speaker)
            
            #    for every_speaker in speaker_labels:
            #        j = j + 1
            #        txtResultFile.write("{0},{1}".format(np.int_(speaker),np.int_(every_speaker)))
            #        if i==len(speaker_ids):
            #            if j<len(speaker_labels):
            #                txtResultFile.write(",")
            #        else:
            #            txtResultFile.write(",")
            
        plt.scatter(reduced_data[speaker_labels,0],
                    reduced_data[speaker_labels,1],
                    color=next(colorcycler))



#plt.plot(reduced_data[:,0], reduced_data[:,1], 'k.',markersize=2)
#plt.plot(reduced_data[:,0],reduced_data[:,1],'g^', reduced_data[:,0])

#plot the centroids as white X
        centroids = kmeans.cluster_centers_
        plt.scatter(centroids[:,0],centroids[:,1],
                    marker='x', s=169, linewidths=3,
                    color='w', zorder=10)
            
        plt.title('K-means clustering on the speakers (PCA-reduced data)')
            
        plt.xlim(x_min,x_max)
        plt.ylim(y_min,y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()




###Method
def cluster(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    
    estimated_labels = estimator.predict(data)
    
    
    homogeneity_score = metrics.homogeneity_score(labels,estimated_labels)
    completeness_score = metrics.completeness_score(labels, estimated_labels)
    v_measure_score = metrics.v_measure_score(labels, estimated_labels)
    adjusted_rand_score = metrics.adjusted_rand_score(labels, estimated_labels)
    adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(labels, estimated_labels)
    #    silhouette_score = metrics.silhouette_score(features,  estimated_labels,
    #                                                metric='euclidean',
    #                                                sample_size=sample_size)
    
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

    txtFinalResultFile.write("{0}".format(fileID))
    txtFinalResultFile.write(",")
    txtFinalResultFile.write("{0}".format(completeness_score))
    txtFinalResultFile.write("\n");

    print('Name: % 9s \n'
          'Time: %.2fs \n'
          'Homogeneity Score: %.3f \n'
          'Completeness Score: %.3f \n'
          'V Measure score: %.3f \n'
          'Adjusted rand score: %.3f \n'
          'Adjusted Mutual Info score: %.3f \n'
          % (name, (time()-t0),
             homogeneity_score,
             completeness_score,
             v_measure_score,
             adjusted_rand_score,
             adjusted_mutual_info_score))

    return completeness_score


for fileID in range(fileIDMin,fileIDMax+1):

    ###Prepare output file path
    outputRoot = "/Users/avrosh/Documents/Coursework/7100_Spring_16/Dataset/dataset/"+set+"/"+"set"+set+"_S"+numSpeakers+"_"+hopLength+"_"+blockLength+"_"+str(fileID)+"_"+thresholdOrder
    if extraid != 0:
        outputRoot = outputRoot + "_" + str(extraid)
    outputRoot = outputRoot + "_" + clusterType + ".csv"
    # print outputRoot

    txtResultFile = open(outputRoot, "w")

    ###Prepare input file path
    path = "/Users/avrosh/Documents/Coursework/7100_Spring_16/Dataset/dataset/"+set+"/features/set"+set+"_"+hopLength+"_"+blockLength+"_S"+numSpeakers+"_"+str(fileID)+"_"+thresholdOrder
    if extraid != 0:
        path = path + "_" + str(extraid)
    path = path + ".csv"
    #print path

    f = open(path)
    f.readline()

    ###Read data
    data = np.loadtxt(fname = f, delimiter=',')

    all_labels = data[:,0]
    labels = all_labels[all_labels != 0]
    #labels = data[:,0]
    #print labels

    #normalize data
    #features = scale(data[:,1:])
    features = data[data[:,0] != 0]
    features = scale(features[:,1:])
    #features = data[:,1:]
    #print features


    n_samples, n_features = features.shape
    n_speakers = len(np.unique(labels))
    speaker_ids = np.unique(labels)
    print speaker_ids
    print ("n_speakers %d \nn_samples %d \nn_features %d" % (n_speakers,n_samples,n_features))

    sample_size = 300

    print(79 * '_')
    print fileID

    #KMeans
    if (clusterType == "kmeans"):
        overall_completeness_score = overall_completeness_score + cluster(KMeans(init='k-means++', n_clusters=n_speakers, n_init=10),
                name='k-means++',
                data=features)
#        visualize_kmeans(features)

    ##KMeans with random initialization
    if (clusterType == "kmeans-rand"):
        overall_completeness_score = overall_completeness_score + cluster(KMeans(init='random', n_clusters=n_speakers, n_init=10),
                name='Random',
                data=features)
#        visualize_kmeans(features)

    #
    ##KMeans PCA
    #in this case the seeding of the centers in deterministic, hence we run the algorithm only once
    if (clusterType == "kmeans-pca"):
        pca = PCA(n_components=n_speakers).fit(features)
        overall_completeness_score = overall_completeness_score + cluster(KMeans(init=pca.components_, n_clusters=n_speakers, n_init=1),
                name='PCA-based',
                data=features)
#        visualize_kmeans(features)


    ##GMM
    # Fit a mixture of Gaussians with EM using five components
    if (clusterType == "gmm"):
        gmm = mixture.GMM(n_components=n_speakers, covariance_type='full')
        overall_completeness_score = overall_completeness_score + cluster(gmm,
                name='gmm',
                data=features)
#        visualize_gmm(features,gmm)


    fileCount = fileCount + 1;

    ###Close output file
    txtResultFile.close()

txtFinalResultFile.write("{0}".format("Mean"))
txtFinalResultFile.write(",")
txtFinalResultFile.write("{0}".format(overall_completeness_score/fileCount))

print overall_completeness_score/fileCount

txtFinalResultFile.close()















