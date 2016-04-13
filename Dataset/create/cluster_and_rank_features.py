import sys
from time import time
import pandas as pd
from pandas import DataFrame 
import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed(42)

set = sys.argv[1];              #Set
numSpeakers = sys.argv[2];      #Number of Speakers
blockLength = sys.argv[3];      #Block length
hopLength = sys.argv[4];        #Hop length
fileID = sys.argv[5];
#fileIDMin = sys.argv[5];        #fileID - lower limit
#fileIDMax = sys.argv[6];        #fileID - higher limit

#scores_kmeans = np.array([1,2,3,4,5,6,7])
scores_kmeans = np.array([0])
#scores_rand = np.array([1,2,3,4,5,6,7])

#########################################################################


def bench_k_means(estimator, name, data):
    
    t0 = time()
    estimator.fit(data)
    
    homogeneity_score = metrics.homogeneity_score(labels,estimator.labels_)
    completeness_score = metrics.completeness_score(labels, estimator.labels_)
    v_measure_score = metrics.v_measure_score(labels, estimator.labels_)
    adjusted_rand_score = metrics.adjusted_rand_score(labels, estimator.labels_)
    adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(labels, estimator.labels_)
    silhouette_score = metrics.silhouette_score(features,  estimator.labels_,
                                                metric='euclidean',
                                                sample_size=sample_size)
    
#    return np.array([(time()-t0),homogeneity_score,completeness_score,v_measure_score,adjusted_rand_score,adjusted_mutual_info_score,silhouette_score])

    return np.array([completeness_score])

#        print('Name: % 9s \n'
#              'Time: %.2fs \n'
#              'Estimator Inertia: %i \n'
#              'Homogeneity Score: %.3f \n'
#              'Completeness Score: %.3f \n'
#              'V Measure score: %.3f \n'
#              'Adjusted rand score: %.3f \n'
#              'Adjusted Mutual Info score: %.3f \n'
#              'Silhouette Score: %.3f'
#              % (name, (time()-t0),estimator.inertia_,
#                 homogeneity_score,
#                 completeness_score,
#                 v_measure_score,
#                 adjusted_rand_score,
#                 adjusted_mutual_info_score,
#                 silhouette_score))

def visualize_kmeans(feature_vector):
    #Visualize data
    reduced_data = PCA(n_components=2).fit_transform(feature_vector)
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
    colorcycler = itertools.cycle(['r', 'g', 'b', 'y','b','w','c','m'])

    for speaker in speaker_ids:
        speaker_labels = np.argwhere(labels==speaker)
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
#########################################################################

#path = "/Users/avrosh/Documents/Coursework/7100_Spring_16/Dataset/dataset/"+set+"/features/set"+set+"_"+hopLength+"_"+blockLength+"_S"+numSpeakers+"_"+fileID+"_allfeatures.csv"

#path = "/Users/avrosh/Documents/Coursework/7100_Spring_16/Dataset/dataset/"+set+"/features/set"+set+"_"+hopLength+"_"+blockLength+"_S"+numSpeakers+"_"+fileID+".csv"

#path = "/Users/avrosh/Documents/Coursework/7100_Spring_16/Dataset/dataset/"+set+"/features/set"+set+"_"+hopLength+"_"+blockLength+"_S"+numSpeakers+"_"+fileID+"_onlyMFCCs.csv"

path = "/Users/avrosh/Documents/Coursework/7100_Spring_16/Dataset/dataset/"+set+"/features/set"+set+"_"+hopLength+"_"+blockLength+"_S"+numSpeakers+"_"+fileID+"_selectedfeatures.csv"




f = open(path)
headers = f.readline().strip().split(",")

data = np.loadtxt(fname = f, delimiter=',')

labels = data[:,0]
#print labels

#features = scale(data[:,1:])
features = data[:,1:]
#print features

#
n_samples, n_features = features.shape
n_speakers = len(np.unique(labels))
speaker_ids = np.unique(labels)
print speaker_ids
print ("n_speakers %d \nn_samples %d \nn_features %d" % (n_speakers,n_samples,n_features))


for x in xrange(0,n_features):
#for x in xrange(0,1):


    feature_vector = features[:,x].reshape(-1, 1)
    
    #repeat if you want to visualize the data
#    feature_vector = np.repeat(feature_vector,2,axis=1)

#    print feature_vector.shape
#    print x

    sample_size = 300
    
#    print feature_vector

    #KMeans
    scores_kmeans = np.vstack((scores_kmeans, bench_k_means(KMeans(init='k-means++', n_clusters=n_speakers, n_init=10), name='k-means++', data=feature_vector)))


#    #KMeans with random initialization
#    bench_k_means(KMeans(init='random', n_clusters=n_speakers, n_init=10),
#                  name='Random',
#                  data=feature_vector)

#    #KMeans PCA
#    #in this case the seeding of the centers in deterministic, hence we run the algorithm only once
#    pca = PCA(n_components=n_speakers).fit(feature_vector)
#    paramsKMeansPCA[x] = bench_k_means(KMeans(init=pca.components_, n_clusters=n_speakers, n_init=1),
#                  name='PCA-based',
#                  data=feature_vector)
#    print feature_vector.shape


#    visualize_kmeans(feature_vector)

    print(79 * '_')

print "And the ranked features are: "
#scores = scores[:,1:]

scores_kmeans = scores_kmeans[1:]

print scores_kmeans

rank_indices = np.argsort(scores_kmeans,axis=0)

headers = headers[1:]

for i in xrange(n_features-1,-1,-1):
    print headers[rank_indices[i]]


#print headers[np.argsort(scores_kmeans)]

#print scores_kmeans



#print('Name: % 9s \n'
#      'Time: %.2fs %s \n'
#      'Estimator Inertia: %i %s \n'
#      'Homogeneity Score: %.3f %s \n'
#      'Completeness Score: %.3f %s \n'
#      'V Measure score: %.3f %s \n'
#      'Adjusted rand score: %.3f %s \n'
#      'Adjusted Mutual Info score: %.3f %s \n'
#      'Silhouette Score: %.3f %s'
#      % ("Kmeans", paramsKMeans.time_to_process_min,headers[paramsKMeans.time_to_process_min_ind + 1],
#         paramsKMeans.estimator_intertia_max,headers[paramsKMeans.estimator_intertia_max_ind + 1],
#         paramsKMeans.homogeneity_score_max,headers[paramsKMeans.homogeneity_score_max_ind + 1],
#         paramsKMeans.completeness_score_max,headers[paramsKMeans.completeness_score_max_ind + 1],
#         paramsKMeans.v_measure_score_max,headers[paramsKMeans.v_measure_score_max_ind + 1],
#         paramsKMeans.adjusted_rand_score_max,headers[paramsKMeans.adjusted_rand_score_max_ind + 1],
#         paramsKMeans.adjusted_mutual_info_score_max,headers[paramsKMeans.adjusted_mutual_info_score_max_ind + 1],
#         paramsKMeans.silhouette_score_max,headers[paramsKMeans.silhouette_score_max_ind + 1]))





















