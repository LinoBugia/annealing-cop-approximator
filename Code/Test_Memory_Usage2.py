from Funcs_Optimizers import *

import os
import psutil

## Function to get current memory usage
def current_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss  # Resident Set Size - memory used by the process

## Check memory before creating a large object

print("starting feature generation")
start_time_gen =time.time() 
num_MC = 10
steps=1000

variables = 200

degree = 2

type_alg = "SimulatedAnnealing"

Offset_increase = 10

memory_vardict = []
memory_pbf = []
numVars = []

variablestep = 50

seed_gen = 76243
seed_rand =[]
random.seed(seed_gen)
aprr= 10
aprr2 = 3000
apprlist=[]
for _ in range(aprr):
    time.sleep(5)
    variable =[]
    time.sleep(5)
    initial_memory = current_memory_usage()
    for _ in range(aprr2):
        variable.append(float(random.random()))
    time.sleep(5)
    memory_after = current_memory_usage()
    print("Memory used by large object:", memory_after - initial_memory)
    apprlist.append(memory_after - initial_memory)


average = np.array(apprlist).sum()/(aprr2*aprr)

print("average ="+str(average))
if 0:
    for i in range(35):
        if 1:
            time.sleep(5)
            pbf_var_dict = dict()
            pbf = dict()
            initial_memory = current_memory_usage()
            pbf = createPoly(variables,degree,1,seed=seed_gen)
            time.sleep(5)
            memory_after = current_memory_usage()
            print("Memory used by large object:", memory_after - initial_memory)
            memory_pbf.append(memory_after - initial_memory)
            initial_memory = current_memory_usage()
            pbf_var_dict = createPolyDict(pbf,variables) 
            time.sleep(5)
            memory_after = current_memory_usage()
            print("Memory used by large object:", memory_after - initial_memory)
            memory_vardict.append(memory_after - initial_memory)
            numVars.append(variables)

            variables += variablestep

    fig = make_subplots(rows=2,cols=1,
                    subplot_titles=["memory_pbf","memory_vardict"]
                    )

    fig.add_trace(go.Scatter(x=numVars, y=memory_pbf,showlegend=False),row=1,col=1)
    fig.add_trace(go.Scatter(x=numVars, y=memory_vardict,showlegend=False),row=2,col=1)
    fig.show()