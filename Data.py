"""This file enables to create objects associated to the datasets. The clouds -- who will represent
 the different dataset -- are lists of Point objects."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Point:

    def __init__(self,coordinates:tuple,basis_ : np.ndarray|list = None):
        """
        Create an object point from its coordinates in a specific basis
        ### Parameters
        :param tuple coordinates : coordinates of the point
        :param ArrayLike basis_ : orthonormal base associated to the point for the previous coordinates
        """
        coords=coordinates
        dim=len(coordinates)
        #if the basis is not precised, we assume it is the canonical basis
        if basis_==None:
            match self.dim:
                case 1:
                    basis=[np.array([1.])]
                case 2:
                    basis=[np.array([1.,0.]),np.array([0.,1.])]
                case 3:
                    basis=[np.array([1.,0.,0.]),np.array([0.,1.,0.]),np.array([0.,0.,1.])]

        else:
            basis=basis_
    
    def _get_canonical_basis(self):
        """
        Gives the canonical basis corresponding to the dimension of the space

        :return: canonical basis corresponding to the dimension of the space
        """
        match self.dim:
            case 1:
                return [np.array([1.])]
            case 2:
                return [np.array([1.,0.]),np.array([0.,1.])]
            case 3:
                return [np.array([1.,0.,0.]),np.array([0.,1.,0.]),np.array([0.,0.,1.])]

    def change_basis(self,new_basis : list|np.ndarray):
        """
        Gives the coordinates of the point in a new orthonormal basis 
        ### Parameter
        :param Arraylike new_basis : contains an orthonormal base (each vector being of type numpy Array), 
        which we want to use for the point
        :return: replace the former coordinates by the coordinates in the new basis
        """
        
        basis=self._get_canonical_basis()
        vector=self.coords[0]*self.basis[0]
        for i in range(1,self.dim):
            vector+=self.coords[i]*self.basis[i]
        for i in range(len(new_basis)):
            self.coords[i]=np.dot(vector,new_basis[i])
        self.basis=new_basis

    def print(self):
        print("- Dimension :",self.dim)
        print("- Base :",self.basis)
        print("- Coordinates :",self.coords)


class Cloud:

    def __init__(self,data_set:np.ndarray|list|pd.core.frame.DataFrame):
        """
        Create an object associated with a group of points
        ### Parameters
        :param ArrayLike|Dataframe data_set : initial represention of the points
        """
        length=len(data_set)
        points=[]
        if type(data_set) == pd.core.frame.DataFrame:
            data=data_set.iloc
        else:
            data=data_set
        for ind in range(length):
            pnt=data[ind]
            points.append(Point(tuple(pnt)))

    def get_point(self,i:int):
        """
        Gets the ith point of the data set
        ### Parameter
        :param int i : index of the point we want to get
        :return: a point
        :rtype: Point
        """
        if i>=0 and i<self.length:
            return self.points[i]
    
    def add_point(self,coords:tuple,basis_ : np.ndarray|list = None):
        """
        Adds a point in the data set

        :param int dim : dimension of the new point
        :param coord_x : the x coordinate (float)
        :param  coord_y,coord_z (optional) : the x and y coordinates
        :param Arraylike basis : the basis associated to the coordinates
        :return: a point
        :rtype: Point
        """
        self.points.append(Point(tuple,basis=basis_))
        self.length+=1
    
    def remove_point(self,i:int):
        """
        Removes the ith point of the data set and returns it
        ### Parameter
        :param int i : index of the point we want to remove
        :return: the place at the ith place
        :rtype: Point
        """
        if i>=0 and i<self.length:
            self.length -=1
            return self.points.pop(i)
    
    def print(self,title:str="Display of the dataset",rotation:list|np.ndarray|tuple=[30.,30.,30.]):
        """
        Displays the dataset
        ### Parameters
        :param str title : title of the plot
        :param Arraylike rotation : contains the angle used to display the dataset (if 3-dimensional)
        :return: a plot
        """
        match self.dim:
            case 1: 
                X=[],Y=[]
                for i in range(self.length):
                    pnt=self.get_point(i)
                    X.append(pnt[0])
                    Y.append(0.)
                plt.figure()
                plt.scatter(X,Y)
                plt.xlabel("x axis")
            case 2: 
                X=[],Y=[]
                for i in range(self.length):
                    pnt=self.get_point(i)
                    X.append(pnt[0])
                    Y.append(pnt[1])
                plt.figure()
                plt.scatter(X,Y)
                plt.xlabel("x axis")
                plt.ylabel("y axis")
            case 3: 
                X=[],Y=[],Z=[]
                for i in range(self.length):
                    pnt=self.get_point(i)
                    X.append(pnt[0])
                    Y.append(pnt[1])
                    Z.append(pnt[2])
                plt.figure()
                plt.scatter(X,Y,Z)
                plt.xlabel("x axis")
                plt.ylabel("y axis")
                plt.ylabel("y axis")
    plt.show()