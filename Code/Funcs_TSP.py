#functions related to TSP

# functions to generate TSP problems from real data + conversion functions
from itertools import combinations
import numpy as np
from Funcs_Annealing2 import *
import random
import pandas as pd
#import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Some locations
from Funcs_Annealers import DigitalAnnealing

def plot_tour(coords):

    fig_intermediate = make_subplots(rows=1,cols=1)
    fig_intermediate.update_layout(title_text="Route")
    fig_intermediate.update_layout(
    autosize=False,
    width=800,
    height=800,
    xaxis=dict(scaleanchor="y", scaleratio=1),
    yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    x,y=np.hsplit(np.array(coords),2)                          
    x = x.flatten()
    y = y.flatten()
    fig_intermediate.add_trace(
    go.Scatter(
        mode="lines+markers" ,
        showlegend=True,
        x=x,
        y=y),row=1,col=1
        )
    fig_intermediate.show()

def Tour_Clean_Up_Routeswitch(tour,Matrix):
    combs=[]
    for i in range(len(tour)):
        if i == len(tour)-1:
            combs.append((tour[len(tour)-1],tour[0]))
        else:
            combs.append((tour[i],tour[i+1]))
    
    combs2 = list(combinations(range(len(combs)),2))
    for comb2 in combs2:
        dist1=Matrix[combs[comb2[0]]]+Matrix[combs[comb2[1]]]
        dist2=Matrix[(combs[comb2[0]][0],combs[comb2[1]][0])]+Matrix[(combs[comb2[0]][1],combs[comb2[1]][1])]
        if  dist1 > dist2:
            tour[comb2[0]],tour[comb2[1]] = tour[comb2[1]],tour[comb2[0]]
            break
        
    return tour

def read_in_csv_TSP(path):
    """
    This function reads in a TSP_problem from a path
    returns:
        coords
    """


    df = pd.read_csv(path,skiprows=6,names=["name","x","y"],delim_whitespace=True)
    df = df.drop("name",axis = "columns")
    df = df[:-1].astype("float32")

    coords = df.to_numpy()
    return coords

def shortest_path_heuristic(Matrix,starting_point=1):

    """
    This procedure implements the shortest path heuristic
    returns:
        shortest path from starting_point as beginning
    """
    Matrix = Matrix + Matrix.transpose()
    
    Tour = []
    options = list(range(np.shape(Matrix)[0]))
    
    #TODO add random select
    Tour.append(starting_point)
    options.remove(starting_point)
    for i in range(np.shape(Matrix)[0]):
        dist = 0
        for j in options:
            if Matrix[i,j]>dist:
                target = j
                options.remove(j)
                Tour.append(target)

    return Tour



#make the dict to represent the matrix representation
def convert_Matrix_to_pbf(Matrix:np.array,penalty_term:bool,a=1,b=1):
    """
    Converts a matrix of distances to the TSP problem
    returns:
        pbf
        translation dict
        inverse translation dict
    """
    Matrix = Matrix + Matrix.transpose()

    N=Matrix.shape[1]
    trans_dict = dict()
    row = -1
    for var in range(N*N):
        column = var%N
        if column == 0:
            row = row +1
        trans_dict[var]= (row,column)

    inv_trans_dict =dict(zip(trans_dict.values(), trans_dict.keys()))
    # Hamiltonian circle condition
    pbf = dict()
    combs = list(combinations(range(N),2))
    if penalty_term:
        A= np.max(Matrix)*a
        pbf[()]= N*A          #korrekt
        for j in range(N):
            for i in range(N):
                pbf[(inv_trans_dict[(j,i)],)] = -2*A    #korrekt
        for j in range(N):            
            for comb in combs:
                #if pbf.__contains__((inv_trans_dict[(j,comb[0])],inv_trans_dict[(j,comb[1])])):
                #    pbf[(inv_trans_dict[(j,comb[0])],inv_trans_dict[(j,comb[1])])]+=2*A
                if 1:
                    pbf[(inv_trans_dict[(j,comb[0])],inv_trans_dict[(j,comb[1])])]=2*A
        for j in range(N):    
            for comb in combs:
                #print((inv_trans_dict[(comb[0]),j],inv_trans_dict[(comb[1],j)]))
                if 1:
                    pbf[(inv_trans_dict[(comb[0],j)],inv_trans_dict[(comb[1],j)])]=2*A
    if 1:
        # Distances cond
        B = b*1
        for comb in combs:
            for j in range(N+1):
                #pbf[(inv_trans_dict[((j+1)%N,comb[0])],inv_trans_dict[(j%N,comb[1])])]=B*Matrix[comb] #hin und rück richtung
                #pbf[(inv_trans_dict[((j+1)%N,comb[1])],inv_trans_dict[(j%N,comb[0])])]=B*Matrix[comb] 

                pbf[(inv_trans_dict[(j%N,comb[0])],inv_trans_dict[((j+1)%N,comb[1])])]=B*Matrix[comb]
                pbf[(inv_trans_dict[(j%N,comb[1])],inv_trans_dict[((j+1)%N,comb[0])])]=B*Matrix[comb] 
    return pbf, trans_dict, inv_trans_dict

def make_random_coords(x_1,x_2,y_1,y_2,Size,seed = 42):
    """
    Makes a random Matrix of distances in the range x_1 - x_2 and y_1 - y_2
    returns:
        coords
    """
    random.seed(seed)
    coords = []
    for i in range(Size):
        coords.append([random.uniform(x_1,x_2),random.uniform(y_1,y_2)])
    return coords

def calculate_dist_Matrix(coords:list[list[float,float]]):
    """"""
    coordpairs = list(combinations(range(len(coords)),2))
    Matrix = np.zeros((len(coords),len(coords)))
    for coordpair in coordpairs:    
        vektor = np.array([coords[list(coordpair)[0]][0]-coords[list(coordpair)[1]][0],coords[list(coordpair)[0]][1]-coords[list(coordpair)[1]][1]])
        Matrix[list(coordpair)[0],list(coordpair)[1]]= np.linalg.norm(vektor)
    return Matrix

def Convert_var_Assignment_to_order(varAssignement,trans_dict)->list[int]:
    solution = []
    for i in range(len(varAssignement)):
        if varAssignement[i]==1:
            solution.append([trans_dict[i][0],trans_dict[i][1]])

    solution = sorted(solution,key = lambda x:solution[1])
    x,y=np.hsplit(np.array(solution),2)
    return y.flatten()

def GroupA_TransferClosestPointGroupB(GroupA,GroupB):
    min = 1000
    loc=-1
    point = []
    for pointA in GroupA:
        i=0
        for pointB in GroupB:
            dist = np.sqrt(float(pointA[0]-pointB[0])**2+float(pointA[1]-pointB[1])**2)
            if dist < min:
                point = pointB
                min = dist
                loc=i
            i=i+1
    if loc >-1:
        GroupA.append(point)
        del GroupB[loc]
    return GroupA,GroupB

def Connect_2_Tours_Closest_point(GroupA,GroupB):
    min = 1000000000000000
    j=0
    for pointA in GroupA:
        i=0
        j=j+1
        for pointB in GroupB:
            dist = np.sqrt(float(pointA[0]-pointB[0])**2+float(pointA[1]-pointB[1])**2)
            if dist < min:
                locA=j
                locB=i
                min = dist
            i=i+1
    Group = []
    for i in range(len(GroupA)):
        if i != locA-1:
            Group.append(GroupA[i])
        else: 
            Group.append(GroupA[locA-1])
            for j in range(len(GroupB)):
                Group.append(GroupB[locB-j])

    return Group

def plot_solution(coords,orders,red):
    fig = make_subplots(rows=1,cols=1)
    for j in range(len(orders)):
        if np.mod(j,red)==0:
            ordered_coords=[]
            for i in orders[j]:
                ordered_coords.append(coords[i])
            x,y=np.hsplit(np.array(ordered_coords),2)
            x = x.flatten()
            y = y.flatten()
            fig.add_trace(
                go.Scatter(
                    visible=False,
                    mode="lines+markers" ,
                    name= str(j),
                    x=x,
                    y=y)
                    )

    # Make 10th trace visible
    fig.data[0].visible = True
    
    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        order = orders[int(i*red)]
        dist = 0
        for j in range(1,len(order)):
            vektor = np.array([coords[order[j]][0]-coords[order[j-1]][0],coords[order[j]][1]-coords[order[j-1]][1]])
            dist += np.linalg.norm(vektor)
        vektor = np.array([coords[order[len(order)-1]][0]-coords[order[0]][0],coords[order[len(order)-1]][1]-coords[order[0]][1]])
        dist += np.linalg.norm(vektor)
        text = "calculated distance: " +str(dist)
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                {"title": text}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Frequency: "},
        pad={"t": 1},
        steps=steps
    )]
    fig.update_layout(
        sliders=sliders
    )
    fig.show()


    
    #plot_solution(coords,orders)

    # Beispielprobleme 
    # https://plotly.com/python/sliders/
    # http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
    # https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2014.00005/full