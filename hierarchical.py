import random
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from matplotlib.mlab import PCA as mlabPCA
from sklearn.decomposition import PCA as sklearnPCA
from scipy.spatial.distance import euclidean
import heapq

print " "
file_name = raw_input("Enter file name:")
print " "

main_matrix = np.loadtxt(file_name, dtype = "string", delimiter = '\t')
gene_id = main_matrix[:,0]
ground_truth = main_matrix[:,1]
all_attributes = main_matrix[:,2:]
all_attributes = all_attributes.astype(np.float)

number_of_ids = all_attributes.shape[0]  #number of rows
number_of_attributes = all_attributes.shape[1]   #number of columns

ground_truth = ground_truth.astype(np.int)
unique_clusters = np.unique(ground_truth)

print " "
k = raw_input("Enter number of clusters:")
print " "
#k = len(unique_clusters)
k = int(k)

distance_matrix = np.empty((number_of_ids,number_of_ids))

dict_cluster_sets = {}

for i in range(number_of_ids):
    dict_cluster_sets[i] = i

heap = []
for i in range(number_of_ids):
    for j in range(number_of_ids):
        distance_matrix[i][j] = euclidean(all_attributes[i], all_attributes[j])
for i in range(number_of_ids):
    for j in range(i + 1,number_of_ids ):
        two_clusters = []
        two_clusters.append(i)
        two_clusters.append(j)
        heapq.heappush(heap, (distance_matrix[i][j], two_clusters))

unique_vals = len(np.unique(dict_cluster_sets.values()))
while(unique_vals > k):
    min_dist = heapq.heappop(heap)
    if(min_dist[1][0] < min_dist[1][1]):
        min_cluster = min_dist[1][0]
        max_cluster = min_dist[1][1]
    else:
        min_cluster = min_dist[1][1]
        max_cluster = min_dist[1][0]

    val_of_min = dict_cluster_sets.get(min_cluster)
    val_of_max = dict_cluster_sets.get(max_cluster)
    dict_cluster_sets[max_cluster] = val_of_min

    for key, value in dict_cluster_sets.items():
        if (value == val_of_max):
            dict_cluster_sets[key] = val_of_min


    unique_vals = len(np.unique(dict_cluster_sets.values()))

u = np.unique(dict_cluster_sets.values())


for key,value in dict_cluster_sets.items():
    itemindex = np.where(u == value)
    dict_cluster_sets[key] = itemindex[0][0]


cluster_values = dict_cluster_sets.values()

ground_incidence = np.empty((len(ground_truth),len(ground_truth)))
for i in range(len(ground_truth)):
    for j in range(len(ground_truth)):
        if(ground_truth[i] == ground_truth[j]):
            ground_incidence[i][j] = 1
        else:
            ground_incidence[i][j] = 0

cluster_incidence = np.empty((len(cluster_values),len(cluster_values)))
for i in range(len(cluster_values)):
    for j in range(len(cluster_values)):
        if(cluster_values[i] == cluster_values[j]):
            cluster_incidence[i][j] = 1
        else:
            cluster_incidence[i][j] = 0

length = np.unique(cluster_values)
number = len(length)

print " "
print "number of clusters: " , number
print " "
results_2 =[]

t = 0
while t < number:
    results_1 =[]
    for i in range(len(cluster_values)):
        if cluster_values[i] == t:
            results_1.append(gene_id[i])
    results_2.append(results_1)
    t = t + 1
print " "
print "the clusters are"
print results_2
print " "

m00 = 0
m01 = 0
m10 = 0
m11 = 0

for i in range(len(cluster_values)):
    for j in range(len(cluster_values)):
        if(ground_incidence[i][j] == cluster_incidence[i][j] == 1):
            m11 = m11 + 1
        elif(ground_incidence[i][j] == cluster_incidence[i][j] == 0):
            m00 = m00 + 1
        elif(cluster_incidence[i][j] == 1 and ground_incidence[i][j] == 0):
            m10 = m10 + 1
        elif(cluster_incidence[i][j] == 0 and ground_incidence[i][j] == 1):
            m01 = m01 + 1

jaccard = float(m11)/float(m11 + m10 + m01)
rand = float(m11+m00)/float(m11 + m10 + m01+m00)
print "jaccard coefficient" , jaccard
print " "
print "rand coefficient" , rand
print " "

#Implement PCA on dataset and plot clusters

string_pca = "PCA visualization for "
pca = string_pca + file_name

sklearn_pca = sklearnPCA(n_components=2)
X_r = sklearn_pca.fit_transform(all_attributes)

x_val = X_r[:,0]
y_val = X_r[:,1]
plt.scatter(x_val, y_val, c = cluster_values, label="test")
plt.title(pca)

plt.grid(True)

plt.show()
