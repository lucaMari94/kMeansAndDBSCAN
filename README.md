# Exercizes on clustering

Program to read a data file in csv format

The data file has the following header: number of samples, number of characteristics, list of characteristics (comma separated)

Lines contain one example per line.
For each row there is a list of real numbers (separated by commas) which are the values of the characteristics.


```python
import csv
import sys
from os.path import join

import numpy as np
 
# this function reads the data file, loads the configuration attributes specifiefd in the heading
# (numer of examples and features), the list of feature names
# and loads the data in a matrix named data    
def load_data(file_path, file_name):
   with open(join(file_path, file_name)) as csv_file:
       data_file = csv.reader(csv_file,delimiter=',')
       temp1 = next(data_file)
       n_samples = int(temp1[0])
       n_features = int(temp1[1])
       temp2 = next(data_file)
       feature_names = np.array(temp2[:n_features])

       data_list = [iter for iter in data_file]
               
       data = np.asarray(data_list, dtype=np.float64)                  
       
   return(data, feature_names, n_samples, n_features)

# The main program reads the input file containing the dataset
# file_path is the file path where the file with the data to be read are located
# we assume the file contains an example per line
# each example is a list of real values separated by a comma (csv format)
# The first line of the file contains the heading with:
# N_samples,n_features,
# The second line contains the feature names separated by a comma     

#file_path="content/"
file_path="./datasets/"
# all the three datasets contain data points on (x,y) 
file_name1="3-clusters.csv"
file_name2="dataset-DBSCAN.csv"     
file_name3="CURE-complete.csv"    
data1,feature_names1,n_samples1,n_features1 = load_data(file_path, file_name1)
data2,feature_names2,n_samples2,n_features2 = load_data(file_path, file_name2)
data3,feature_names3,n_samples3,n_features3 = load_data(file_path, file_name3)
print("dataset n. 1: ", file_name1, " - n samples = ", n_samples1, " - n features = ", n_features1)
print("dataset n. 1: ", file_name2, " - n samples = ", n_samples2, " - n features = ", n_features2)
print("ddataset n. 1: ", file_name3, " - n samples = ", n_samples3, " - n features = ", n_features3)
```

    dataset n. 1:  3-clusters.csv  - n samples =  150  - n features =  2
    dataset n. 1:  dataset-DBSCAN.csv  - n samples =  6118  - n features =  2
    ddataset n. 1:  CURE-complete.csv  - n samples =  86558  - n features =  2
    

The following program plots the dataset n.1


```python
import matplotlib.pyplot as plt

x = np.linspace(0.0, 10.0, 1000)
y = np.sin(x)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(18, 10))

#plot the datasets
# dataset1: 3-clusters.csv
ax[0,0].plot(data1[:,0], data1[:,1], '.', markersize=5)
ax[0,0].set_title('Dataset n. 1 of data points: 3-clusters.csv')
ax[0,0].set_xlabel(feature_names1[0])
ax[0,0].set_ylabel(feature_names1[1])

# dataset2: dataset-DBSCAN.csv
ax[0,1].plot(data2[:,0], data2[:,1], '.', markersize=2)
ax[0,1].set_title('Dataset n. 2 of data points: dataset-DBSCAN.csv')
ax[0,1].set_xlabel(feature_names2[0])
ax[0,1].set_ylabel(feature_names2[1])

# dataset3: CURE-complete.csv
ax[1,0].plot(data3[:,0], data3[:,1], '.', markersize=2)
ax[1,0].set_title('Dataset n. 3 of data points: CURE-complete.csv')
ax[1,0].set_xlabel(feature_names3[0])
ax[1,0].set_ylabel(feature_names3[1])

plt.show()
```


    
![png](output_4_0.png)
    


## In the following program we cluster the dataset n.1 with K-means. 
From the plot of dataset n.1 we see 3 separated clusters. Thus k=3. 


```python
import csv
import sys
from os.path import join

import numpy as np

file_path="./datasets/"
file_name1="3-clusters.csv"

data1, feature_names1, n_samples1, n_features1 = load_data(file_path, file_name1)

from sklearn.cluster import KMeans

np.random.seed(5)

k=3
kmeans1 = KMeans(n_clusters=k, random_state=0).fit(data1)

i=0
for i in range(n_samples1):
    print("Example n."+str(i)+"=("+str(data1[i,0])+","+str(data1[i,1])+") in cluster n."+str(kmeans1.labels_[i]))
```

    Example n.0=(10.0,10.0) in cluster n.0
    Example n.1=(11.0,11.0) in cluster n.0
    Example n.2=(12.0,12.0) in cluster n.0
    Example n.3=(13.0,13.0) in cluster n.0
    Example n.4=(14.0,14.0) in cluster n.0
    Example n.5=(15.0,15.0) in cluster n.0
    Example n.6=(16.0,16.0) in cluster n.0
    Example n.7=(17.0,17.0) in cluster n.0
    Example n.8=(18.0,18.0) in cluster n.0
    Example n.9=(19.0,19.0) in cluster n.0
    Example n.10=(20.0,20.0) in cluster n.0
    Example n.11=(21.0,21.0) in cluster n.0
    Example n.12=(22.0,22.0) in cluster n.0
    Example n.13=(23.0,23.0) in cluster n.0
    Example n.14=(24.0,24.0) in cluster n.0
    Example n.15=(25.0,25.0) in cluster n.0
    Example n.16=(26.0,26.0) in cluster n.0
    Example n.17=(27.0,27.0) in cluster n.0
    Example n.18=(28.0,28.0) in cluster n.0
    Example n.19=(29.0,29.0) in cluster n.0
    Example n.20=(30.0,30.0) in cluster n.0
    Example n.21=(31.0,31.0) in cluster n.0
    Example n.22=(32.0,32.0) in cluster n.0
    Example n.23=(33.0,33.0) in cluster n.0
    Example n.24=(34.0,34.0) in cluster n.0
    Example n.25=(35.0,35.0) in cluster n.0
    Example n.26=(36.0,36.0) in cluster n.0
    Example n.27=(37.0,37.0) in cluster n.0
    Example n.28=(38.0,38.0) in cluster n.0
    Example n.29=(39.0,39.0) in cluster n.0
    Example n.30=(40.0,40.0) in cluster n.0
    Example n.31=(41.0,41.0) in cluster n.0
    Example n.32=(42.0,42.0) in cluster n.0
    Example n.33=(43.0,43.0) in cluster n.0
    Example n.34=(44.0,44.0) in cluster n.0
    Example n.35=(45.0,45.0) in cluster n.0
    Example n.36=(46.0,46.0) in cluster n.0
    Example n.37=(47.0,47.0) in cluster n.0
    Example n.38=(48.0,48.0) in cluster n.0
    Example n.39=(49.0,49.0) in cluster n.0
    Example n.40=(50.0,50.0) in cluster n.0
    Example n.41=(51.0,51.0) in cluster n.0
    Example n.42=(52.0,52.0) in cluster n.0
    Example n.43=(53.0,53.0) in cluster n.0
    Example n.44=(54.0,54.0) in cluster n.0
    Example n.45=(55.0,55.0) in cluster n.0
    Example n.46=(56.0,56.0) in cluster n.0
    Example n.47=(57.0,57.0) in cluster n.0
    Example n.48=(58.0,58.0) in cluster n.0
    Example n.49=(80.0,80.0) in cluster n.2
    Example n.50=(81.0,81.0) in cluster n.2
    Example n.51=(82.0,82.0) in cluster n.2
    Example n.52=(83.0,83.0) in cluster n.2
    Example n.53=(84.0,84.0) in cluster n.2
    Example n.54=(85.0,85.0) in cluster n.2
    Example n.55=(86.0,86.0) in cluster n.2
    Example n.56=(87.0,87.0) in cluster n.2
    Example n.57=(88.0,88.0) in cluster n.2
    Example n.58=(89.0,89.0) in cluster n.2
    Example n.59=(90.0,90.0) in cluster n.2
    Example n.60=(91.0,91.0) in cluster n.2
    Example n.61=(92.0,92.0) in cluster n.2
    Example n.62=(93.0,93.0) in cluster n.2
    Example n.63=(94.0,94.0) in cluster n.2
    Example n.64=(95.0,95.0) in cluster n.2
    Example n.65=(96.0,96.0) in cluster n.2
    Example n.66=(97.0,97.0) in cluster n.2
    Example n.67=(98.0,98.0) in cluster n.2
    Example n.68=(99.0,99.0) in cluster n.2
    Example n.69=(100.0,100.0) in cluster n.2
    Example n.70=(101.0,101.0) in cluster n.2
    Example n.71=(102.0,102.0) in cluster n.2
    Example n.72=(103.0,103.0) in cluster n.2
    Example n.73=(104.0,104.0) in cluster n.2
    Example n.74=(105.0,105.0) in cluster n.2
    Example n.75=(106.0,106.0) in cluster n.2
    Example n.76=(107.0,107.0) in cluster n.2
    Example n.77=(108.0,108.0) in cluster n.2
    Example n.78=(109.0,109.0) in cluster n.2
    Example n.79=(110.0,110.0) in cluster n.2
    Example n.80=(111.0,111.0) in cluster n.2
    Example n.81=(112.0,112.0) in cluster n.2
    Example n.82=(113.0,113.0) in cluster n.2
    Example n.83=(114.0,114.0) in cluster n.2
    Example n.84=(115.0,115.0) in cluster n.2
    Example n.85=(116.0,116.0) in cluster n.2
    Example n.86=(117.0,117.0) in cluster n.2
    Example n.87=(118.0,118.0) in cluster n.2
    Example n.88=(119.0,119.0) in cluster n.2
    Example n.89=(120.0,120.0) in cluster n.2
    Example n.90=(121.0,121.0) in cluster n.2
    Example n.91=(122.0,122.0) in cluster n.2
    Example n.92=(123.0,123.0) in cluster n.2
    Example n.93=(124.0,124.0) in cluster n.2
    Example n.94=(125.0,125.0) in cluster n.2
    Example n.95=(126.0,126.0) in cluster n.2
    Example n.96=(127.0,127.0) in cluster n.2
    Example n.97=(128.0,128.0) in cluster n.2
    Example n.98=(129.0,129.0) in cluster n.2
    Example n.99=(120.0,10.0) in cluster n.1
    Example n.100=(121.0,11.0) in cluster n.1
    Example n.101=(122.0,12.0) in cluster n.1
    Example n.102=(123.0,13.0) in cluster n.1
    Example n.103=(124.0,14.0) in cluster n.1
    Example n.104=(125.0,15.0) in cluster n.1
    Example n.105=(126.0,16.0) in cluster n.1
    Example n.106=(127.0,17.0) in cluster n.1
    Example n.107=(128.0,18.0) in cluster n.1
    Example n.108=(129.0,19.0) in cluster n.1
    Example n.109=(130.0,20.0) in cluster n.1
    Example n.110=(131.0,21.0) in cluster n.1
    Example n.111=(132.0,22.0) in cluster n.1
    Example n.112=(133.0,23.0) in cluster n.1
    Example n.113=(134.0,24.0) in cluster n.1
    Example n.114=(135.0,25.0) in cluster n.1
    Example n.115=(136.0,26.0) in cluster n.1
    Example n.116=(137.0,27.0) in cluster n.1
    Example n.117=(138.0,28.0) in cluster n.1
    Example n.118=(139.0,29.0) in cluster n.1
    Example n.119=(140.0,30.0) in cluster n.1
    Example n.120=(141.0,31.0) in cluster n.1
    Example n.121=(142.0,32.0) in cluster n.1
    Example n.122=(143.0,33.0) in cluster n.1
    Example n.123=(144.0,34.0) in cluster n.1
    Example n.124=(145.0,35.0) in cluster n.1
    Example n.125=(146.0,36.0) in cluster n.1
    Example n.126=(147.0,37.0) in cluster n.1
    Example n.127=(148.0,38.0) in cluster n.1
    Example n.128=(149.0,39.0) in cluster n.1
    Example n.129=(150.0,40.0) in cluster n.1
    Example n.130=(151.0,41.0) in cluster n.1
    Example n.131=(152.0,42.0) in cluster n.1
    Example n.132=(153.0,43.0) in cluster n.1
    Example n.133=(154.0,44.0) in cluster n.1
    Example n.134=(155.0,45.0) in cluster n.1
    Example n.135=(156.0,46.0) in cluster n.1
    Example n.136=(157.0,47.0) in cluster n.1
    Example n.137=(158.0,48.0) in cluster n.1
    Example n.138=(159.0,49.0) in cluster n.1
    Example n.139=(160.0,50.0) in cluster n.1
    Example n.140=(161.0,51.0) in cluster n.1
    Example n.141=(162.0,52.0) in cluster n.1
    Example n.142=(163.0,53.0) in cluster n.1
    Example n.143=(164.0,54.0) in cluster n.1
    Example n.144=(165.0,55.0) in cluster n.1
    Example n.145=(166.0,56.0) in cluster n.1
    Example n.146=(167.0,57.0) in cluster n.1
    Example n.147=(168.0,58.0) in cluster n.1
    Example n.148=(169.0,59.0) in cluster n.1
    Example n.149=(170.0,60.0) in cluster n.1
    

In the following program we plot the clusters


```python
import matplotlib.pyplot as plt

plt.style.use('default')
fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(111)
fig.subplots_adjust(top=1)
ax.set_title('Clustered points in dataset n. 1')

ax.set_xlabel('x')
ax.set_ylabel('y')

# set the list of colors to be selected when plotting the different clusters
color=['b','y','g','k','m','c','r','w']
    
#plot the dataset
for clu in range(k):
    # collect the sequence of cooordinates of the points in each given cluster (determined by clu)
    data_list_x = [data1[i,0] for i in range(n_samples1) if kmeans1.labels_[i]==clu]
    data_list_y = [data1[i,1] for i in range(n_samples1) if kmeans1.labels_[i]==clu]
    plt.scatter(data_list_x, data_list_y, s=20, edgecolors='none', c=color[clu], alpha=0.5)

plt.show()

```


    
![png](output_8_0.png)
    


# In the following program we cluster the dataset n.2 with K-means.
From the plot of dataset n.2 we see 3 separated clusters. Thus k=3
The best value of k depends on how the data is distributed.


```python
import csv
import sys
from os.path import join

import numpy as np

file_path="./datasets/"
file_name2="dataset-DBSCAN.csv"
data2,feature_names2,n_samples2,n_features2 = load_data(file_path, file_name2)

from sklearn.cluster import KMeans

np.random.seed(5)

k=3
kmeans2 = KMeans(n_clusters=k, random_state=0).fit(data2)

i=0
for i in range(n_samples2):
    print("Example n."+str(i)+"=("+str(data2[i,0])+","+str(data2[i,1])+") in cluster n."+str(kmeans1.labels_[i]))
```

    Example n.0=(5.0,482.0) in cluster n.0
    Example n.1=(5.0,481.0) in cluster n.0
    Example n.2=(5.0,480.0) in cluster n.0
    Example n.3=(5.0,479.0) in cluster n.0
    Example n.4=(5.0,478.0) in cluster n.0
    Example n.5=(5.0,477.0) in cluster n.0
    Example n.6=(5.0,476.0) in cluster n.0
    Example n.7=(5.0,475.0) in cluster n.0
    Example n.8=(5.0,474.0) in cluster n.0
    Example n.9=(5.0,473.0) in cluster n.0
    Example n.10=(5.0,472.0) in cluster n.0
    Example n.11=(5.0,471.0) in cluster n.0
    Example n.12=(5.0,470.0) in cluster n.0
    Example n.13=(5.0,469.0) in cluster n.0
    Example n.14=(5.0,468.0) in cluster n.0
    Example n.15=(5.0,467.0) in cluster n.0
    Example n.16=(5.0,466.0) in cluster n.0
    Example n.17=(5.0,465.0) in cluster n.0
    Example n.18=(5.0,464.0) in cluster n.0
    Example n.19=(5.0,463.0) in cluster n.0
    Example n.20=(5.0,462.0) in cluster n.0
    Example n.21=(5.0,461.0) in cluster n.0
    Example n.22=(5.0,460.0) in cluster n.0
    Example n.23=(6.0,482.0) in cluster n.0
    Example n.24=(6.0,481.0) in cluster n.0
    Example n.25=(6.0,480.0) in cluster n.0
    Example n.26=(6.0,479.0) in cluster n.0
    Example n.27=(6.0,478.0) in cluster n.0
    Example n.28=(6.0,477.0) in cluster n.0
    Example n.29=(6.0,476.0) in cluster n.0
    Example n.30=(6.0,475.0) in cluster n.0
    Example n.31=(6.0,474.0) in cluster n.0
    Example n.32=(6.0,473.0) in cluster n.0
    Example n.33=(6.0,472.0) in cluster n.0
    Example n.34=(6.0,471.0) in cluster n.0
    Example n.35=(6.0,470.0) in cluster n.0
    Example n.36=(6.0,469.0) in cluster n.0
    Example n.37=(6.0,468.0) in cluster n.0
    Example n.38=(6.0,467.0) in cluster n.0
    Example n.39=(6.0,466.0) in cluster n.0
    Example n.40=(6.0,465.0) in cluster n.0
    Example n.41=(6.0,464.0) in cluster n.0
    Example n.42=(6.0,463.0) in cluster n.0
    Example n.43=(6.0,462.0) in cluster n.0
    Example n.44=(6.0,461.0) in cluster n.0
    Example n.45=(6.0,460.0) in cluster n.0
    Example n.46=(7.0,482.0) in cluster n.0
    Example n.47=(7.0,481.0) in cluster n.0
    Example n.48=(7.0,480.0) in cluster n.0
    Example n.49=(7.0,479.0) in cluster n.2
    Example n.50=(7.0,478.0) in cluster n.2
    Example n.51=(7.0,477.0) in cluster n.2
    Example n.52=(7.0,476.0) in cluster n.2
    Example n.53=(7.0,475.0) in cluster n.2
    Example n.54=(7.0,474.0) in cluster n.2
    Example n.55=(7.0,473.0) in cluster n.2
    Example n.56=(7.0,472.0) in cluster n.2
    Example n.57=(7.0,471.0) in cluster n.2
    Example n.58=(7.0,470.0) in cluster n.2
    Example n.59=(7.0,469.0) in cluster n.2
    Example n.60=(7.0,468.0) in cluster n.2
    Example n.61=(7.0,467.0) in cluster n.2
    Example n.62=(7.0,466.0) in cluster n.2
    Example n.63=(7.0,465.0) in cluster n.2
    Example n.64=(7.0,464.0) in cluster n.2
    Example n.65=(7.0,463.0) in cluster n.2
    Example n.66=(7.0,462.0) in cluster n.2
    Example n.67=(7.0,461.0) in cluster n.2
    Example n.68=(7.0,460.0) in cluster n.2
    Example n.69=(8.0,490.0) in cluster n.2
    Example n.70=(8.0,489.0) in cluster n.2
    Example n.71=(8.0,488.0) in cluster n.2
    Example n.72=(8.0,487.0) in cluster n.2
    Example n.73=(8.0,486.0) in cluster n.2
    Example n.74=(8.0,485.0) in cluster n.2
    Example n.75=(8.0,484.0) in cluster n.2
    Example n.76=(8.0,483.0) in cluster n.2
    Example n.77=(8.0,482.0) in cluster n.2
    Example n.78=(8.0,481.0) in cluster n.2
    Example n.79=(8.0,480.0) in cluster n.2
    Example n.80=(8.0,479.0) in cluster n.2
    Example n.81=(8.0,478.0) in cluster n.2
    Example n.82=(8.0,477.0) in cluster n.2
    Example n.83=(8.0,476.0) in cluster n.2
    Example n.84=(8.0,475.0) in cluster n.2
    Example n.85=(8.0,474.0) in cluster n.2
    Example n.86=(8.0,473.0) in cluster n.2
    Example n.87=(8.0,472.0) in cluster n.2
    Example n.88=(8.0,471.0) in cluster n.2
    Example n.89=(8.0,470.0) in cluster n.2
    Example n.90=(8.0,469.0) in cluster n.2
    Example n.91=(8.0,468.0) in cluster n.2
    Example n.92=(8.0,467.0) in cluster n.2
    Example n.93=(8.0,466.0) in cluster n.2
    Example n.94=(8.0,465.0) in cluster n.2
    Example n.95=(8.0,464.0) in cluster n.2
    Example n.96=(8.0,463.0) in cluster n.2
    Example n.97=(8.0,462.0) in cluster n.2
    Example n.98=(8.0,461.0) in cluster n.2
    Example n.99=(8.0,460.0) in cluster n.1
    Example n.100=(8.0,459.0) in cluster n.1
    Example n.101=(8.0,458.0) in cluster n.1
    Example n.102=(8.0,457.0) in cluster n.1
    Example n.103=(8.0,456.0) in cluster n.1
    Example n.104=(9.0,490.0) in cluster n.1
    Example n.105=(9.0,489.0) in cluster n.1
    Example n.106=(9.0,488.0) in cluster n.1
    Example n.107=(9.0,487.0) in cluster n.1
    Example n.108=(9.0,486.0) in cluster n.1
    Example n.109=(9.0,485.0) in cluster n.1
    Example n.110=(9.0,484.0) in cluster n.1
    Example n.111=(9.0,483.0) in cluster n.1
    Example n.112=(9.0,482.0) in cluster n.1
    Example n.113=(9.0,481.0) in cluster n.1
    Example n.114=(9.0,480.0) in cluster n.1
    Example n.115=(9.0,479.0) in cluster n.1
    Example n.116=(9.0,478.0) in cluster n.1
    Example n.117=(9.0,477.0) in cluster n.1
    Example n.118=(9.0,476.0) in cluster n.1
    Example n.119=(9.0,475.0) in cluster n.1
    Example n.120=(9.0,474.0) in cluster n.1
    Example n.121=(9.0,473.0) in cluster n.1
    Example n.122=(9.0,472.0) in cluster n.1
    Example n.123=(9.0,471.0) in cluster n.1
    Example n.124=(9.0,470.0) in cluster n.1
    Example n.125=(9.0,469.0) in cluster n.1
    Example n.126=(9.0,468.0) in cluster n.1
    Example n.127=(9.0,467.0) in cluster n.1
    Example n.128=(9.0,466.0) in cluster n.1
    Example n.129=(9.0,465.0) in cluster n.1
    Example n.130=(9.0,464.0) in cluster n.1
    Example n.131=(9.0,463.0) in cluster n.1
    Example n.132=(9.0,462.0) in cluster n.1
    Example n.133=(9.0,461.0) in cluster n.1
    Example n.134=(9.0,460.0) in cluster n.1
    Example n.135=(9.0,459.0) in cluster n.1
    Example n.136=(9.0,458.0) in cluster n.1
    Example n.137=(9.0,457.0) in cluster n.1
    Example n.138=(9.0,456.0) in cluster n.1
    Example n.139=(10.0,490.0) in cluster n.1
    Example n.140=(10.0,489.0) in cluster n.1
    Example n.141=(10.0,488.0) in cluster n.1
    Example n.142=(10.0,487.0) in cluster n.1
    Example n.143=(10.0,486.0) in cluster n.1
    Example n.144=(10.0,485.0) in cluster n.1
    Example n.145=(10.0,484.0) in cluster n.1
    Example n.146=(10.0,483.0) in cluster n.1
    Example n.147=(10.0,482.0) in cluster n.1
    Example n.148=(10.0,481.0) in cluster n.1
    Example n.149=(10.0,480.0) in cluster n.1
    


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_22436\3412030360.py in <module>
         18 i=0
         19 for i in range(n_samples2):
    ---> 20     print("Example n."+str(i)+"=("+str(data2[i,0])+","+str(data2[i,1])+") in cluster n."+str(kmeans1.labels_[i]))
    

    IndexError: index 150 is out of bounds for axis 0 with size 150



```python
import matplotlib.pyplot as plt

plt.style.use('default')
fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(111)
fig.subplots_adjust(top=1)
ax.set_title('Clustered points in dataset n. 2')

ax.set_xlabel('x')
ax.set_ylabel('y')

# set the list of colors to be selected when plotting the different clusters
color=['b','y','g','k','m','c','r','w']
    
#plot the dataset
for clu in range(k):
    # collect the sequence of cooordinates of the points in each given cluster (determined by clu)
    data_list_x = [data2[i,0] for i in range(n_samples2) if kmeans2.labels_[i]==clu]
    data_list_y = [data2[i,1] for i in range(n_samples2) if kmeans2.labels_[i]==clu]
    plt.scatter(data_list_x, data_list_y, s=20, edgecolors='none', c=color[clu], alpha=0.5)

plt.show()
```


    
![png](output_11_0.png)
    


# As you can see, some points/examples are classified incorrectly.
This is because the clusters have an irregular shape: the clusters have different sizes from each other.
To solve this problem we can think of increasing the number of clusters k.
I try with k = 4.
The best value of k depends on how the data is distributed.


```python
import csv
import sys
from os.path import join

import numpy as np

file_path="./datasets/"
file_name2="dataset-DBSCAN.csv"
data2,feature_names2,n_samples2,n_features2 = load_data(file_path, file_name2)

from sklearn.cluster import KMeans

np.random.seed(5)

k=4
kmeans2 = KMeans(n_clusters=k, random_state=0).fit(data2)

# plot
plt.style.use('default')
fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(111)
fig.subplots_adjust(top=1)
ax.set_title('Clustered points in dataset n. 2')

ax.set_xlabel('x')
ax.set_ylabel('y')

# set the list of colors to be selected when plotting the different clusters
color=['b','y','g','k','m','c','r','w']
    
#plot the dataset
for clu in range(k):
    # collect the sequence of cooordinates of the points in each given cluster (determined by clu)
    data_list_x = [data2[i,0] for i in range(n_samples2) if kmeans2.labels_[i]==clu]
    data_list_y = [data2[i,1] for i in range(n_samples2) if kmeans2.labels_[i]==clu]
    plt.scatter(data_list_x, data_list_y, s=20, edgecolors='none', c=color[clu], alpha=0.5)

plt.show()
```


    
![png](output_13_0.png)
    


# In the following program we cluster the dataset n.3 with K-means.
From the plot of dataset n.2 we see 3 separated clusters. Thus k=3


```python
import csv
import sys
from os.path import join

import numpy as np

file_path="./datasets/"
file_name3="CURE-complete.csv"
data3,feature_names3,n_samples3,n_features3 = load_data(file_path, file_name3)

from sklearn.cluster import KMeans

np.random.seed(5)

k=3
kmeans3 = KMeans(n_clusters=k, random_state=0).fit(data3)

# plot
plt.style.use('default')
fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(111)
fig.subplots_adjust(top=1)
ax.set_title('Clustered points in dataset n. 3')

ax.set_xlabel('x')
ax.set_ylabel('y')

# set the list of colors to be selected when plotting the different clusters
color=['b','y','g','k','m','c','r','w']
    
#plot the dataset
for clu in range(k):
    # collect the sequence of cooordinates of the points in each given cluster (determined by clu)
    data_list_x = [data3[i,0] for i in range(n_samples3) if kmeans3.labels_[i]==clu]
    data_list_y = [data3[i,1] for i in range(n_samples3) if kmeans3.labels_[i]==clu]
    plt.scatter(data_list_x, data_list_y, s=20, edgecolors='none', c=color[clu], alpha=0.5)

plt.show()
```


    
![png](output_15_0.png)
    


# As you can see, some points/examples are classified incorrectly.
This is because the clusters have an irregular shape: the clusters have different sizes from each other.
To solve this problem we can think of increasing the number of clusters k.
I try with k = 5.
The best value of k depends on how the data is distributed.


```python
import csv
import sys
from os.path import join

import numpy as np

file_path="./datasets/"
file_name3="CURE-complete.csv"
data3,feature_names3,n_samples3,n_features3 = load_data(file_path, file_name3)

from sklearn.cluster import KMeans

np.random.seed(5)

k=5
kmeans3 = KMeans(n_clusters=k, random_state=0).fit(data3)

# plot
plt.style.use('default')
fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(111)
fig.subplots_adjust(top=1)
ax.set_title('Clustered points in dataset n. 3')

ax.set_xlabel('x')
ax.set_ylabel('y')

# set the list of colors to be selected when plotting the different clusters
color=['b','y','g','k','m','c','r','w']
    
#plot the dataset
for clu in range(k):
    # collect the sequence of cooordinates of the points in each given cluster (determined by clu)
    data_list_x = [data3[i,0] for i in range(n_samples3) if kmeans3.labels_[i]==clu]
    data_list_y = [data3[i,1] for i in range(n_samples3) if kmeans3.labels_[i]==clu]
    plt.scatter(data_list_x, data_list_y, s=20, edgecolors='none', c=color[clu], alpha=0.5)

plt.show()
```


    
![png](output_17_0.png)
    


# Quantitative evaluation of clusters in the three datasets: Silhouttes 
**Note:**
Execute K-means a certain number of times (let us try 10 times) and then select the clustering solution that gives the best value of the evaluation measure.


```python
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def silhouetteAnalysis(dataset):
    range_n_clusters = [2, 3, 4, 5, 6]
    for n_clusters in range_n_clusters:
       # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(dataset) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, n_init=10, random_state=10)
        cluster_labels = clusterer.fit_predict(dataset)
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(dataset, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(dataset, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            dataset[:, 0], dataset[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )
        
        plt.show()
```

# dataset 1: 3-clusters.csv


```python
silhouetteAnalysis(data1)
```

    For n_clusters = 2 The average silhouette_score is : 0.5651096232579514
    


    
![png](output_21_1.png)
    


    For n_clusters = 3 The average silhouette_score is : 0.7229402453727759
    


    
![png](output_21_3.png)
    


    For n_clusters = 4 The average silhouette_score is : 0.6825234611999134
    


    
![png](output_21_5.png)
    


    For n_clusters = 5 The average silhouette_score is : 0.6358957746571624
    


    
![png](output_21_7.png)
    


    For n_clusters = 6 The average silhouette_score is : 0.6113126050223654
    


    
![png](output_21_9.png)
    


# dataset 2: dataset-DBSCAN.csv


```python
silhouetteAnalysis(data2)
```

    For n_clusters = 2 The average silhouette_score is : 0.43322366812579544
    


    
![png](output_23_1.png)
    


    For n_clusters = 3 The average silhouette_score is : 0.4799133604477169
    


    
![png](output_23_3.png)
    


    For n_clusters = 4 The average silhouette_score is : 0.41736481925553526
    


    
![png](output_23_5.png)
    


    For n_clusters = 5 The average silhouette_score is : 0.42560503070470135
    


    
![png](output_23_7.png)
    


    For n_clusters = 6 The average silhouette_score is : 0.41296908138576005
    


    
![png](output_23_9.png)
    


# dataset 3: CURE-complete.csv


```python
# silhouetteAnalysis(data3)
```

# In the following cell I propose you to run DBSCAN, instead, on one of the last two datasets: either dataset2 or dataset3. 

At the beginning try using a pair of Minpts and Eps of your choice.

**Note:**
If the data is too big, **sample it random, using a factor of 0.1.**

# DBSCAN - dataset-DBSCAN.csv


```python
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt

def plot_dbscan(db,data,n_samples):
    plt.figure()
    plt.title("Cluster")
    # set the list of colors to be selected when plotting the different clusters
    color=['b','y','g','k','m','c','r','w']
        
    #plot the dataset
    k = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    for clu in range(k):
        # collect the sequence of cooordinates of the points in each given cluster (determined by clu)
        data_list_x = [data[i,0] for i in range(n_samples) if db.labels_[i]==clu]
        data_list_y = [data[i,1] for i in range(n_samples) if db.labels_[i]==clu]
        plt.scatter(data_list_x, data_list_y, s=2, edgecolors='none', c=color[clu], alpha=0.5)
        
    plt.show()


eps = 1 # number of points that fall into a portion;
min_pts = 5 # core point density threshold = minimum number of points to make a portion high intensity
db = DBSCAN(eps=eps, min_samples=min_pts).fit(data2)

labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print('Estimated number of core points: %d' % len(db.core_sample_indices_))
print('Estimated number of border points: %d' % (len(labels)-n_noise_))

plot_dbscan(db,data2,n_samples2)

```

    Estimated number of clusters: 6
    Estimated number of noise points: 114
    Estimated number of core points: 4978
    Estimated number of border points: 6004
    


    
![png](output_28_1.png)
    

