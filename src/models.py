"""This file contains two classes, the first containing all the information
about the modelisation of a given wire (pertinent axis, spatial extension and
parameters of the catenary curve).
The seconds enables to create objects of the first one, and also enables to plot
the point cloud and its model."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from math import cosh
from src.Clusters import Clusters
from src.Data import Point

def catenary_curve(x:float|int,x_0:float|int,y_0:float|int,c:float|int):
    """
    Gives the value of the catenary function with parameters
    ### Parameters
    :param scalar x : the variable
    :param scalars x_0,y_0,c : parameters of the catenary function 
    :return: a tuple
    """
    if type(x) in [int,float,np.float64]:
        return(y_0+c*(cosh((x-x_0)/c)-1))
    else:
        y=[]
        for x_i in x:
            y.append(y_0+c*(cosh((x_i-x_0)/c)-1))
        return y

class _3D_CatenaryCurve():
        
    def __init__(
            self,horizontal_vector:np.ndarray,
            normal_vector:np.ndarray,
            vertical_vector:np.ndarray,
            min_horiz:float,max_horiz:float,
            normal_value_:float, parameters_:tuple
                ):
        self.basis=(horizontal_vector,normal_vector,vertical_vector)
        self.horizontal_values=(min_horiz,max_horiz)
        self.normal_value=normal_value_
        self.parameters=parameters_
    
    def generate_points(self,nb_points:int=100):
        """
        Create arrays containing well-distributed points along the
        curve defined in the class
        ### Parameters
        :param int nb_points : number of points we want to generate
        :return : 3 arrays corresponding to the coordinates in the
        canonical basis
        """
        horizontal_values=np.linspace(self.horizontal_values[0],self.horizontal_values[1],
                                      nb_points)
        normal_values=np.array([self.normal_value]*nb_points)
        x_0,y_0,c=self.parameters
        vertical_values=catenary_curve(horizontal_values,x_0,y_0,c)

        x,y,z=[],[],[]

        for i in range(nb_points):
            point=Point((horizontal_values[i],normal_values[i],
                        vertical_values[i]),self.basis)
            point.change_basis([np.identity(3)[0],np.identity(3)[1],np.identity(3)[2]])
            x.append(point.coords[0])
            y.append(point.coords[1])
            z.append(point.coords[2])
        return (x,y,z)
    

class Model(Clusters):

    def __init__(self, data_set: np.ndarray | list | pd.core.frame.DataFrame):
        super().__init__(data_set)
        if not self.clustered:
            self.clustering()
        self.basis=[]
        for cluster in range(self.nb_clusters):
            self.basis.append(self.find_2D_plane(cluster))

        
    def adapted_basis(self):
        """
        Put every point of the dataset in an adapted basis
        """
        for i in range(self.length):
            self.points[i].change_basis(self.basis[self.clusters[i]])
    
    def modelisation_2D(self,cluster):
        """
        Gives the three parameters of the best-fitting caternary curve (x0, y0 and c)
        for the point projected on the horizontal vector and the vertical one.
        ### Parameter
        :param int cluster : the wire we study
        :return: a tuple
        """
        self.adapted_basis()
        horizont_axis=[]
        vertic_axis=[]
        for i in range(self.length):
            coords=self.get_point(i).coords
            if self.clusters[i]==cluster:
                horizont_axis.append(coords[0])
                vertic_axis.append(coords[2])
        parameters_=list(curve_fit(catenary_curve,horizont_axis,vertic_axis,[0.,0.,1.])[0])
        return parameters_
    
    def display_2D_graph(self,cluster):
        """
        Displays a 2D plot of the projection of the cluster on a well-adapted 
        pane as well as the catenary curve assosiated
        ### Parameter
        :param int cluster : the wire we study
        :return: a plot
        """
        self.adapted_basis()
        horizont_axis=[]
        vertic_axis=[]
        for i in range(self.length):
            coords=self.get_point(i).coords
            if self.clusters[i]==cluster:
                horizont_axis.append(coords[0])
                vertic_axis.append(coords[2])
        parameters=self.modelisation_2D(cluster)
        plt.figure()
        plt.scatter(horizont_axis,vertic_axis,c="blue")
        horizont_axis_bis=np.linspace(min(horizont_axis),max(horizont_axis),200)
        print("Parameters : \n")
        print('x_0=',parameters[0])
        print("z_0=",parameters[1])
        print("c=",parameters[2])
        vertic_axis_bis=[catenary_curve(horizont_axis_bis[i],parameters[0],
                                        parameters[1],parameters[2]) 
                                        for i in range(len(horizont_axis_bis))]
        plt.plot(horizont_axis_bis,vertic_axis_bis,c="red",alpha=1.)
        plt.xlabel("Horizontal axis t")
        plt.ylabel("Vertical axis z")
        plt.title("Best fitting catenary curve for the wire "+str(cluster))
        plt.show()

    def _mean_normal_value(self,cluster:int):
        """
        Enables to get the value of the coordinate from the normal vector, which
        is supposed to be orthogonal to the plane associated to the wire.
        """
        self.adapted_basis()
        res=0.
        n=0.
        for i in range(self.length):
            if self.clusters[i]==cluster:
                res+=self.get_point(i).coords[1]
                n+=1.
        return res/max(1.,n)
    
    def modelisation_3D(self,cluster):
        self.adapted_basis()
        horizont_axis=[]
        for i in range(self.length):
            coords=self.get_point(i).coords
            if self.clusters[i]==cluster:
                horizont_axis.append(coords[0])
        return _3D_CatenaryCurve(self.basis[cluster][0],self.basis[cluster][1],
                                 self.basis[cluster][2],min(horizont_axis),
                                 max(horizont_axis),self._mean_normal_value(cluster),
                                 tuple(self.modelisation_2D(cluster)))
    
    def display_3D_graph(self,cluster:int,rotation:tuple=(30,30,30)):
        """
        Displays a 3D plot of the projection of the cluster on a well-adapted 
        pane as well as the catenary curve assosiated
        ### Parameter
        :param int cluster : the wire we study
        :return: a plot
        """
        
        plt.figure()
        x,y,z=[],[],[]
        for i in range(self.length):
            point=self.get_point(i)
            if self.clusters[i]==cluster:
                point.change_basis(np.identity(3))
                x.append(point.coords[0])
                y.append(point.coords[1])
                z.append(point.coords[2])
        ax = plt.subplot(111, projection='3d')
        ax.scatter(x,y,z,color='blue',alpha=0.1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        #ax.view_init(rotation[0],rotation[1],rotation[2])
        plt.title("Best fitting catenary curve for the wire "+str(cluster))

        model=self.modelisation_3D(cluster)
        coords=model.generate_points(100)
        plt.plot(coords[0],coords[1],coords[2],color="red")
        plt.show()