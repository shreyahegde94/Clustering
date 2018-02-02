import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA as mlabPCA
from sklearn.decomposition import PCA as sklearnPCA


print " "
file_name = raw_input("Enter file name:")
print " "

main_matrix = np.loadtxt(file_name, dtype = "string", delimiter = '\t')
gene_id = main_matrix[:,0]
ground_truth = main_matrix[:,1]
all_attributes = main_matrix[:,2:]
all_attributes = all_attributes.astype(np.float)

ground_truth = ground_truth.astype(np.int)

number_of_ids = all_attributes.shape[0]  #number of rows
number_of_attributes = all_attributes.shape[1]   #number of columns

rows = number_of_ids
dataset = all_attributes
data = all_attributes

r = raw_input("Enter maximum distance between two points:")
r = float(r)
print " "
MinPts = raw_input("Enter minimum number of points to make a cluster:")
MinPts = float(MinPts)
print " "

cluster = np.zeros(rows, dtype = "int")

def expandCluster(neighbours, cluster_id):
    for i in neighbours:
        if(cluster[i] == 0):
            cluster[i] = -2
            new_neighbors = region_query(i)
            if(len(new_neighbors) >= MinPts):
                for x in new_neighbors:
                    if x not in neighbours:
                        neighbours.append(x)
        if(cluster[i]== -2):
            cluster[i] = cluster_id
        if(cluster[i] == -1):
            cluster[i] = cluster_id


def region_query(point):
    neighbours = []
    for x in range(0, dataset.shape[0]):
        if(np.linalg.norm(dataset[x] - dataset[point]) < r):
            neighbours.append(x)
    return neighbours


def DBSCAN():
    cluster_id = 1
    for i in range(rows):
        if cluster[i] == 0:
            cluster[i] = -2
            neighbours = region_query(i)
            if(len(neighbours) < MinPts):
                cluster[i] = -1
            else:
                expandCluster(neighbours, cluster_id)
                cluster_id += 1


DBSCAN()
length = np.unique(cluster)
number = len(length)
no_of_clusters = number - 1
print " "
print "number of clusters found excluding the outliers : " , no_of_clusters
print " "
results_2 =[]
results_3 =[]

for i in range(len(cluster)):
    if cluster[i] == -1:
        results_3.append(gene_id[i])

print " "
print "the outliers are"
print results_3

t = 1
while t <= no_of_clusters:
    results_1 =[]
    for i in range(len(cluster)):
        if cluster[i] == t:
            results_1.append(gene_id[i])
    results_2.append(results_1)
    t = t + 1
print " "
print "the clusters are"
print results_2


ground_incidence = np.empty((len(ground_truth),len(ground_truth)))
for i in range(len(ground_truth)):
    for j in range(len(ground_truth)):
        if(ground_truth[i] == ground_truth[j]):
            ground_incidence[i][j] = 1
        else:
            ground_incidence[i][j] = 0

cluster_incidence = np.empty((len(cluster),len(cluster)))
for i in range(len(cluster)):
    for j in range(len(cluster)):
        if(cluster[i] == cluster[j]):
            cluster_incidence[i][j] = 1
        else:
            cluster_incidence[i][j] = 0

m00 = 0
m01 = 0
m10 = 0
m11 = 0

for i in range(len(cluster)):
    for j in range(len(cluster)):
        if(ground_incidence[i][j] == cluster_incidence[i][j] == 1):
            m11 = m11 + 1
        elif(ground_incidence[i][j] == cluster_incidence[i][j] == 0):
            m00 = m00 + 1
        elif(cluster_incidence[i][j] == 1 and ground_incidence[i][j] == 0):
            m10 = m10 + 1
        elif(cluster_incidence[i][j] == 0 and ground_incidence[i][j] == 1):
            m01 = m01 + 1

print " "
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
plt.scatter(x_val, y_val, c = cluster)
plt.title(pca)

plt.grid(True)
plt.show()
