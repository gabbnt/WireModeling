import sys
from matplotlib import pyplot as plt
import pandas as pd

sys.path.append("..")
from src.models import Model

dataset=pd.read_parquet('..\datasets\lidar_cable_points_extrahard.parquet', engine='pyarrow')
dataset=dataset.sort_index(axis = 0, ascending = True)
dataset.dropna(how='any',inplace=True)
dataset.reset_index(drop=True,inplace=True)

model=Model(dataset)
model.clustering()
(model.what_clusters())
for clust in model.what_clusters():
    model.display_3D_graph(clust)
plt.show()
