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

fileID = sys.argv[1];
set = sys.argv[2];
numSpeakers = sys.argv[3];
blockLength = sys.argv[4];
hopLength = sys.argv[5];

#path = "/Users/avrosh/Documents/Coursework/7100_Spring_16/Dataset/dataset/A/features/setA_2048_32000_S4_6.csv"

path = "/Users/avrosh/Documents/Coursework/7100_Spring_16/Dataset/dataset/"+set+"/features/set"+set+"_"+hopLength+"_"+blockLength+"_S"+numSpeakers+"_"+fileID+".csv"


#data = pd.read_csv(path)
#print df.head()

f = open(path)
f.readline()

data = np.loadtxt(fname = f, delimiter=',')

labels = data[:,0]
#print labels

#normalize data
features = scale(data[:,1:])
#features = data[:,1:]
#print features

n_samples, n_features = features.shape
n_speakers = len(np.unique(labels))
speaker_ids = np.unique(labels)
print speaker_ids
print ("n_speakers %d \nn_samples %d \nn_features %d" % (n_speakers,n_samples,n_features))

sample_size = 300

print(79 * '_')

def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('Name: % 9s \n'
          'Time: %.2fs \n'
          'Estimator Inertia: %i \n'
          'Homogeneity Score: %.3f \n'
          'Completeness Score: %.3f \n'
          'V Measure score: %.3f \n'
          'Adjusted rand score: %.3f \n'
          'Adjusted Mutual Info score: %.3f \n'
          'Silhouette Score: %.3f'
          % (name, (time()-t0),estimator.inertia_,
             metrics.homogeneity_score(labels,estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels, estimator.labels_),
             metrics.silhouette_score(features,  estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))


bench_k_means(KMeans(init='k-means++', n_clusters=n_speakers, n_init=10),
              name='k-means++',
              data=features)

print(79 * '_')

bench_k_means(KMeans(init='random', n_clusters=n_speakers, n_init=10),
              name='Random',
              data=features)

print(79 * '_')

#in this case the seeding of the centers in deterministic, hence we run the algorithm only once
pca = PCA(n_components=n_speakers).fit(features)
bench_k_means(KMeans(init=pca.components_, n_clusters=n_speakers, n_init=1),
              name='PCA-based',
              data=features)

    
print(79 * '_')

########################################################################
#Visualize data
reduced_data = PCA(n_components=2).fit_transform(features)
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







