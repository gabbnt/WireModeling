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
                case 1:basis=[np.array([1.])]
                case 2:basis=[np.array([1.,0.]),np.array([0.,1.])]
                case 3:basis=[np.array([1.,0.,0.]),np.array([0.,1.,0.]),np.array([0.,0.,1.])]

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
        :param Arraylike new_basis : contains an orthonormal base (each vector being of type numpy Array), which we want to use for the point
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