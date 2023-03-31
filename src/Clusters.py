"""This file contains the algorithms enabling us to realize a clustering.
We implement a Cluster class (inheriting from the Cloud class), which adds the
notion of clusters and implements a clustering in two parts.
The first clustering algorithm is approximative, and only distinguishs groups of
points that are far away (such as the points in the medium dataset).
The second one uses the fact that the wires are parallel. We can thus realize
a clustering by projecting our points on a well choosen axis."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from src.Data import Cloud

class Clusters(Cloud):

    def __init__(self, data_set: np.ndarray | list | pd.core.frame.DataFrame):
        super().__init__(data_set)
        self.clusters=[-1 for ind in range(self.length)]
        self.nb_clusters=1
        self.clustered=False
    
    def what_clusters(self):
        """
        Determines the different indexes of clusters
        """
        res=[]
        for i in range(self.length):
            clust=self.clusters[i]
            if not clust in res:
                res.append(clust)
        return res

    def _vertical_clustering(self,eps=2.,min_samples=10):
        """
        Realizes at first a rough first clustering (using the DBSCAN algorithm) 
        to split the different groups of wires that are far away from each other 
        (in the case of our datasets, it is only useful for the medium difficulty 
        dataset)
        ### Parameters
        :param float eps (optional) : corresponds to the maximum distance we can 
        allow inside of a cluster
        :param float min_samples (optional) : correspond to the minimum number of 
        points that can form a cluster. Those two default parameters are satisfying 
        enough for this dataset.
        :return: updates the clusters
        """
        points=[]
        for ind in range(self.length):
            point=self.get_point(ind)
            points.append(list(point.coords))

        clusters=DBSCAN(eps=eps,min_samples=min_samples).fit(points)
        self.clusters=list(clusters.labels_)
        self.nb_clusters=max(self.clusters)-min(self.clusters)+1

    def clustering(self,eps:float=0.5,min_samples:int=10,
                   eps_bis:float=2.,min_samples_bis:int=10):
        """
        Clusters the data from the first rough clustering, using a 
        projection on a well choosen line and the DBSCAN algorithm
        ### Parameters
        :param float eps (optional) : corresponds to the maximum distance we can 
        allow inside of a cluster
        :param float min_samples (optional) : correspond to the minimum number of 
        points that can form a cluster. Those two default parameters are satisfying 
        enough for this dataset.
        :param float eps_bis (optional) : same as eps parameter but for the first
        clustering
        :param float min_samples_bis (optional) : same as min_samples parameter but for the
        first clustering
        :return: updates the clusters of self
        """
        self._vertical_clustering(eps_bis,min_samples_bis)
        number_of_vertical_clusters=self.nb_clusters
        clust_to_index=[[] for clust in self.what_clusters()]
        new_clusters=[0 for i in range(self.length)]
        increment_of_cluster_number=0
        for clust in range(number_of_vertical_clusters):
        #First, we figure out the best fitting plane for the whole vertical cluster 
        # (using that all the planes are approximately parallel)
            x,y=[],[]
            for i in range(self.length):
                if self.clusters[i]==clust:
                    clust_to_index[clust].append(i)
                    coordinates=self.get_point(i).coords
                    x.append(coordinates[0])
                    y.append(coordinates[1])
            x=np.array(x)
            y=np.array(y)
            coefficients=np.polyfit(x=x,y=y,deg=1)
            #Then we project the points with a vector normal to the approximated direction
            # of the points
            projections=x*coefficients[0]-y

            #At last we use a clustering algorithm to segregate the different projections
            clusters_i=(DBSCAN(eps=eps,min_samples=min_samples).fit(projections.reshape(-1, 1))).labels_
            for j in range(len(clusters_i)):
                new_clusters[clust_to_index[clust][j]]=clusters_i[j]+increment_of_cluster_number
            increment_of_cluster_number+=max(clusters_i)+1
        self.clusters=new_clusters
        self.nb_clusters=max(new_clusters)-min(new_clusters)+1
        self.clustered=True

    def _find_horizontal_line(self,cluster):
        """
        finds the vector associated to the horizontal part of the wire
        ### Parameters
        :param int cluster : corresponds to the cluster we are interested in
        :return: the horizontal vector characterizing the wire and its normal
        """
        x,y=[],[]
        for i in range(self.length):
            if self.clusters[i]==cluster:
                coordinates=self.get_point(i).coords
                x.append(coordinates[0])
                y.append(coordinates[1])
        coefficients=list(np.polyfit(x=x,y=y,deg=1))

        normal_vector=np.array([coefficients[0],-1.,0.])/np.linalg.norm(
            np.array([coefficients[0],-1.,0.])
        )
        horizontal_vector=np.array([1.,coefficients[0],0.])/np.linalg.norm(
            np.array([1.,coefficients[0],0.])
        )
        return (normal_vector,horizontal_vector)

    def find_2D_plane(self,cluster,lim_corr : float = 0.5):
        """
        Verifies the correlation between the normal axis and the z axis
        to determine wether the z is really the vertical axis or not
        ### Parameters
        :param int cluster : corresponds to the cluster we are interested in
        :param float lim_coor : criteria to know if two coordinates are correlated
        or not
        :return: the horizontal vector, the new normal vector and the new vertical
        vector
        """
        normal_vector,horizontal_vector=self._find_horizontal_line(cluster)
        z_=np.array([0.,0.,1.])
        n,z=[],[]
        for i in range(self.length):
            if self.clusters[i]==cluster:
                point=np.array(list(self.get_point(i).coords))
                n.append(np.dot(point,normal_vector))
                z.append(point[2])
        correlations=np.min(np.corrcoef(n,z))
        if correlations < lim_corr:
            return (horizontal_vector,normal_vector,z_)
        else:
            coefficients_2=list(np.polyfit(x=n,y=z,deg=1))
            new_normal_vector=coefficients_2[0]*normal_vector-z_
            new_normal_vector=new_normal_vector/np.linalg.norm(new_normal_vector)
            new_vertical_vector=coefficients_2[0]*z_+normal_vector
            new_vertical_vector=new_vertical_vector/np.linalg.norm(new_vertical_vector)
            return(horizontal_vector,new_normal_vector,new_vertical_vector)
    
    def print(self,title:str="Display of the dataset",
              rotation:list|np.ndarray|tuple=[30.,30.,30.]):
        """
        Displays the dataset
        ### Parameters
        :param str title : title of the plot
        :param Arraylike rotation : contains the angle used to
        display the dataset (if 3-dimensional)
        :return: a plot, with different colors for different clusters
        """
        colors=["red","blue","green","pink","cyan","purple","yellow"]
        match self.points[0].dim:
            case 1: 
                X,Y=[],[]
                for i in range(self.length):
                    pnt=self.get_point(i)
                    X.append(pnt.coords[0])
                    Y.append(0.)
                plt.figure()
                plt.scatter(X,Y)
                plt.xlabel("x axis")
            case 2: 
                X,Y=[],[]
                for i in range(self.length):
                    pnt=self.get_point(i)
                    X.append(pnt.coords[0])
                    Y.append(pnt.coords[1])
                plt.figure()
                plt.scatter(X,Y)
                plt.xlabel("x axis")
                plt.ylabel("y axis")
            case 3: 
                X,Y,Z=[],[],[]
                for i in range(self.length):
                    pnt=self.get_point(i)
                    X.append(pnt.coords[0])
                    Y.append(pnt.coords[1])
                    Z.append(pnt.coords[2])
                plt.figure()
                ax = plt.subplot(111, projection='3d')
                plt.scatter(X,Y,Z,
                            color=[colors[self.clusters[i]] for i in range(self.length)])
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                ax.view_init(rotation[0],rotation[1],rotation[2])
    plt.show()
