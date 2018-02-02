import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import linalg as LA
from matplotlib.mlab import PCA as mlabPCA
from sklearn.decomposition import PCA as sklearnPCA
from scipy.spatial.distance import euclidean

print " "
file_name = raw_input("Enter file name:")
print " "
print " "
k = raw_input("Enter the number of clusters:")
k = int(k)
print " "
iterations = raw_input("Enter the number of iterations:")
iterations = int(iterations)
print " "
print " "
centers = raw_input("Enter the initial centers:")
print " "

list_centers = []
for i in range(k):
    c = centers.split(",")
new_list = []
for i in range(len(c)):
    x = int(c[i])
    new_list.append(x - 1)
print new_list
print type(new_list[0])


dict_of_points_in_cluster = {}

main_matrix = np.loadtxt(file_name, dtype = "string", delimiter = '\t')
gene_id = main_matrix[:,0]
ground_truth = main_matrix[:,1]
all_attributes = main_matrix[:,2:]
all_attributes = all_attributes.astype(np.float)

ground_truth = ground_truth.astype(np.int)
unique_clusters = np.unique(ground_truth)


number_of_ids = all_attributes.shape[0]  #number of rows
number_of_attributes = all_attributes.shape[1]   #number of columns


rnumber = []
for i in range(len(new_list)):
    ind = new_list[i]
    rnumber.append(all_attributes[ind])


#Pass it the centroid array and all attributes.
#It returns a dictionary with the points in each cluster
def distanceFromCentroids(all_attributes, rnumber):
    list_of_dist = []
    for i in range(k):
        d = euclidean(all_attributes, rnumber[i])
        list_of_dist.append(d)
    cluster_selected = np.argmin(list_of_dist)
    if cluster_selected in dict_of_points_in_cluster:
        dict_of_points_in_cluster[cluster_selected].append(all_attributes)
    else:
        dict_of_points_in_cluster[cluster_selected] = [all_attributes]


#Function to form new centroids
def findNewCentroids(dict_of_clusters):
    new_centroids = []
    for i in dict_of_clusters.keys():
        total = 0;
        no_of_elements = 0;
        for elements in dict_of_clusters[i]:
            no_of_elements = no_of_elements + 1;
            total = total + elements;
        centroid = total/no_of_elements;
        new_centroids.append(centroid)
    return new_centroids

np.apply_along_axis(distanceFromCentroids, 1, all_attributes, rnumber)
key_list = list(dict_of_points_in_cluster.keys())

new_centroids = rnumber

check = True
#while(check):
for i in range(iterations):
    temp_centroids = new_centroids
    new_centroids = findNewCentroids(dict_of_points_in_cluster)
    dict_of_points_in_cluster.clear()
    np.apply_along_axis(distanceFromCentroids, 1, all_attributes, new_centroids)
    allval = 0
    for x in range(k):
        if(np.array_equal(temp_centroids[x] , new_centroids[x])):
            allval = allval + 1

    if(allval == k):
        check = False
#dict_of_points_in_cluster finally has all the points attached to its respective cluster.
#Now we can obtain which rows exactly are attached to which cluster.
revised_dict_with_indices = {}


for key,value in dict_of_points_in_cluster.items():
    for v in value:
        index = np.where((all_attributes == v).all(axis = 1))
        #print "Row index of value ", index[0][0]
        if(len(index[0]) == 1):

            if key in revised_dict_with_indices:
                revised_dict_with_indices[key].append(index[0][0])
            else:
                revised_dict_with_indices[key] = [index[0][0]]
        else:
            length_1 = len(index[0])
            for i in range(length_1):


                if key in revised_dict_with_indices:
                    revised_dict_with_indices[key].append(index[0][i])
                else:
                    revised_dict_with_indices[key] = [index[0][i]]


for key,value in revised_dict_with_indices.items():
    print "Cluster ", key
    print "Elements in cluster ", value

cluster_values = np.zeros(len(ground_truth))
for key,value in revised_dict_with_indices.items():
    for v in value:

        cluster_values[v] = key


#Create incidence matrix based on ground truth and k-means clustering and then calculate Jaccard coefficient


ground_incidence = np.empty((len(ground_truth),len(ground_truth)))
for i in range(len(ground_truth)):
    for j in range(len(ground_truth)):
        if(ground_truth[i] == ground_truth[j]):

            ground_incidence[i][j] = 1
        else:
            ground_incidence[i][j] = 0

cluster_values = np.array(cluster_values, dtype = 'int')


cluster_incidence = np.zeros(shape=(cluster_values.shape[0],cluster_values.shape[0]))
for i in range(cluster_values.shape[0]):
    for j in range(cluster_values.shape[0]):
        if(int(cluster_values[i]) == int(cluster_values[j])):

            cluster_incidence[i][j] = 1
        else:

            cluster_incidence[i][j] = 0

a = 0
d =0
for i in range(len(cluster_values)):
    for j in range(len(cluster_values)):

        if(ground_incidence[i][j] == 1 and (ground_incidence[i][j] == cluster_incidence[i][j])):
            a = a + 1
        if(not((ground_incidence[i][j] == 0) and (cluster_incidence[i][j] == 0))):
            d = d + 1

jaccard = float(a)/float(d)
#rand = float(m11+m00)/float(m11 + m10 + m01+m00)
print " "
print "jaccard coefficient" , jaccard
print " "

#Implement PCA on dataset and plot clusters

string_pca = "PCA visualization for "
pca = string_pca + file_name

sklearn_pca = sklearnPCA(n_components=2)
X_r = sklearn_pca.fit_transform(all_attributes)
#print "X_r", X_r
x_val = X_r[:,0]
y_val = X_r[:,1]
plt.scatter(x_val, y_val, c = cluster_values)
plt.title(pca)

plt.grid(True)
plt.show()
