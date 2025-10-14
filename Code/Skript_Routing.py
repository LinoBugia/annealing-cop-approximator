from routingpy import Valhalla
from pprint import pprint
from itertools import combinations
import numpy as np
import pandas as pd

if 0:   #get locations of real places
    coords = []
    # Some locs in Rgbg
    coords.append([49.017234, 12.098131][::-1])
    coords.append([49.018523, 12.096395][::-1])
    coords.append([49.020045, 12.094231][::-1])
    coords.append([49.017170, 12.083890][::-1])
    coords.append([49.014934, 12.077987][::-1])
    coords.append([49.005722, 12.078465][::-1])
    coords.append([48.981062, 12.122576][::-1])
    coords.append([49.004318, 12.103436][::-1])
    coords.append([49.004176, 12.084310][::-1])
    coords.append([49.025951, 12.108127][::-1])
    coords.append([49.026282, 12.090228][::-1])
    coords.append([49.011751, 12.067637][::-1])
    coords.append([49.016485, 12.113179][::-1])
    coords.append([49.026093, 12.108705][::-1])
    coords.append([49.028033, 12.089795][::-1])
    coords.append([49.012698, 12.143565][::-1])
    coords.append([48.997074, 12.069658][::-1])
    coords.append([49.014450, 12.061863][::-1])

    client = Valhalla()
    #route = client.directions(locations=coords, profile='pedestrian')
    isochrones = client.isochrones(locations=coords[0], profile='pedestrian', intervals=[600, 1200])
    #pprint((route.geometry, route.duration, route.distance, route.raw))
    #pprint((isochrones.raw, isochrones[0].geometry, isochrones[0].center, isochrones[0].interval))
    Time_Matrix=np.zeros((len(coords),len(coords)))
    Distance_Matrix =np.zeros((len(coords),len(coords)))
    coordpairs = list(combinations(range(len(coords)),2))

    for coordpair in coordpairs:
        funcinput=[coords[list(coordpair)[0]], coords[list(coordpair)[1]]]
        matrix = client.matrix(locations=funcinput, profile='pedestrian')
        Time_Matrix[list(coordpair)[0],list(coordpair)[1]] = matrix.durations[0][1]
        Distance_Matrix[list(coordpair)[0],list(coordpair)[1]] = matrix.distances[0][1]
        #print((matrix.durations, matrix.distances))

    print(Time_Matrix)
    print(Distance_Matrix)
    Matrix = np.concatenate((Time_Matrix,Distance_Matrix))
    Matrixa_as_pd = pd.DataFrame(Matrix)
    Matrixa_as_pd.to_csv("Data_4.csv")

    Time_Matrix = Time_Matrix.transpose() +Time_Matrix
    Distance_Matrix = Distance_Matrix.transpose() + Distance_Matrix
    N= len(coords)

if 1: #convert Matrix csv to Dataframe
    Matrix_as_pd= np.array(pd.read_csv("Data_4.csv"))
    N=Matrix_as_pd.shape[1]-1
    Time_Matrix = Matrix_as_pd[0:N,1:].transpose() + Matrix_as_pd[0:N,1:]
    Distance_Matrix = Matrix_as_pd[N:,1:].transpose() + Matrix_as_pd[N:,1:]