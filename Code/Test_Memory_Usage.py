# These are evaluation Scripts

from Funcs_Annealing2 import *
from Funcs_Annealers import *

import time
import datetime
import pandas as pd
import csv
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
def testPerformance():
    variables = 100
    degree = 2
    Num_MC = 10
    Temp = 100
    pbf = createPoly(variables,degree,0.6)
    steps=3000
    #pbf = {(1,): 1, (2,): -1}
    pbf_var_dict = createPolyDict(pbf,variables) 
    #print("PBF: ", pbf)
    if 0:
        start_time = time.time()
        vals = getPBFLandscape(pbf)
        print("Landscape: ", vals)
        print("Minimum: ", np.min(vals))
        print("Execution time: %s seconds" % (time.time() - start_time))
    if 1:
        start_time = time.time()
        val = [simulatedAnnealing(pbf,pbf_var_dict,steps, Temp, initialvarAssignment=[])[0] for i in range(1, Num_MC)]
        #print(val)
        print("Min simulated solution: ", np.min(val))
        print("Max simulated solution: ", np.max(val))
        print("Execution time: %s seconds" % (time.time() - start_time))
    if 1:
        start_time =time.time()
        val3 = [DigitalAnnealing(pbf,pbf_var_dict, steps, Temp, initialvarAssignment=[])[0] for i in range(1, Num_MC)]

        #print(val2)
        print("Min Digital solution: ", np.min(val3))
        print("Max Digital solution: ", np.max(val3))
        print("Execution time normal DA: %s seconds" % (time.time() - start_time))
    if 1:
        start_time =time.time()
        val3 = [DigitalAnnealing2(pbf,pbf_var_dict, steps, Temp, initialvarAssignment=[])[0] for i in range(1, Num_MC)]

        #print(val2)
        print("Min Digital solution: ", np.min(val3))
        print("Max Digital solution: ", np.max(val3))
        print("Execution time paralell DA: %s seconds" % (time.time() - start_time))
    if 0:
        start_time =time.time()
        H2 = ConvertPBFtoTensorFormFull(pbf,variables,degree = degree)
        val2 = [DigitalAnnealing_QUBO(H2[0],H2[1],pbf, steps, Temp, i)[0] for i in range(1, Num_MC)]
        #print(val2)
        print("Min Digital solution: ", np.min(val2))
        print("Max Digital solution: ", np.max(val2))
        print("Execution time: %s seconds" % (time.time() - start_time))


def CompareEvaluations(seed):
    degree = 2
    variables =10
    seed  =97756
    pbf = createPoly(variables,degree,1,seed=seed)
    varAssignement = getInitialVarAssignement(pbf, seed)
    random.seed(seed)
    H = ConvertPBFtoTensorForm(pbf,variables,degree = degree)
    #for i in range(len(H)):
    #    print(H[i])
    #print(pbf)
    varAssignement_vek=ConvertDicToVektor(varAssignement)
    #print(varAssignement_vek)
    start_time = time.time()
    former_energy1 = evalPBF(pbf, varAssignement)
    Energy1=[]
    #TODO hier ist etwas rausgefallen
    print("Execution time Eval Lukas: %s seconds" % (time.time() - start_time))
    #print(Energy1)
    start_time = time.time()
    former_energy2 = EvalPbfAsTensor(H, varAssignement_vek)
    Energy2=[]
    for pos in range(len(varAssignement_vek)):
        varAssignement_vek_c = varAssignement_vek.copy()
        if varAssignement_vek_c[pos]==1:
            varAssignement_vek_c[pos]=0
        else:
            varAssignement_vek_c[pos]=1
        Energy2.append(former_energy2 - EvalPbfAsTensor(H, varAssignement_vek_c))
    print("Execution time Eval As Tensor: %s seconds" % (time.time() - start_time))

    start_time = time.time()
    deltaE1=[]
    for pos in range(len(varAssignement_vek)):
        deltaE1.append(Eval_Delta_Energy(pbf,varAssignement,pos))
    print("Execution time Eval Delta E Lukas: %s seconds" % (time.time() - start_time))
    start_time = time.time()
    deltaE2=[]
    #print(varAssignement_vek)
    H2 = ConvertPBFtoTensorFormFull(pbf,variables,degree = degree)
    #or i in range(len(H2)):
    #    print(H2[i])
    print("Execution time convert: %s seconds" % (time.time() - start_time))
    start_time = time.time()
    for pos in range(len(varAssignement_vek)):
        deltaE2.append(EvalDeltaEAsTensor(H2,varAssignement_vek,pos))

    print("Execution time delta E Tensor: %s seconds" % (time.time() - start_time))
    start_time = time.time()
    deltaE3=Eval_delta_E_as_QUBO(H2[0],H2[1],varAssignement_vek)
    print("Execution time Eval QUBO form: %s seconds" % (time.time() - start_time))
    #print(deltaE3)
    start_time = time.time()
    deltaE4=Eval_delta_E_as_PUBO(H2,varAssignement_vek)
    print("Execution time eval delta E PUBO form: %s seconds" % (time.time() - start_time))
    #print(deltaE4)
    Results=pd.DataFrame()
    Results["var"]=varAssignement_vek
    Results["list compl"]=Energy1
    Results["tensor compl"]=Energy2
    Results["part list"]=deltaE1
    Results["part tensor"]=deltaE2
    Results["QUBO"]=deltaE3
    Results["PUBO"]=deltaE4
    print(Results)


def Giga_testPerformance():
    #make File
    steps=5000
    Num_MC = 10
    Temp = 1000
    
    df = pd.DataFrame(list())
    filename = 'Evaluation_'+str(datetime.date.today())+'.csv'
    df.to_csv(filename)
    filler_row = [""]
    with open(filename , 'a') as csvfile:
        csvwriter =csv.writer(csvfile)
        csvwriter.writerow(["Evaluation: " +str(datetime.date.today())])
        csvwriter.writerow(filler_row)
        csvwriter.writerow(["metaparameters:","steps = " +str(steps), "Num_MC = "+str(Num_MC),"Temp = " +str(Temp)])
        csvwriter.writerow(filler_row)
        csvwriter.writerow(["Type","Time","Min","vars","degree","seed"])
        csvfile.close()
    min_grad = 1
    max_grad = 5
    grad_step = 1
    grads = [min_grad +grad_step*i for i in range(0, int(max_grad/grad_step)-1)]
    min_variables=50
    max_variables=800
    variables_step=150
    variables = [min_variables +variables_step*i for i in range(0, int(max_variables/variables_step))]

    seeds= [1435643,7234952,92134213,5134223,897773]
    
    #Evaluate one after the other 
    #SA
    for gr in grads:
        for var in variables:
            for seed in seeds:
                #set up variables
                pbf = createPoly(var,gr,1,seed=seed)
                pbf_var_dict = createPolyDict(pbf,var) 

                #Method 1: SA

                start_time = time.time()
                val = [simulatedAnnealing(pbf,pbf_var_dict,steps, Temp, i)[0] for i in range(1, Num_MC)]
                Exec_time = time.time() - start_time
                print(gr,var,seed)
                print("Min SA: ", np.min(val))
                print("Max SA: ", np.max(val))
                print("Execution time: %s seconds" % (Exec_time))
                with open(filename, "a") as csvfile:
                    csvwriter =csv.writer(csvfile)
                    csvwriter.writerow(["SA",Exec_time,np.min(val),var,gr,seed])

                #Method 2: DA
                start_time = time.time()
                val = [DigitalAnnealing(pbf,pbf_var_dict,steps, Temp, i)[0] for i in range(1, Num_MC)]
                Exec_time = time.time() - start_time
                print(gr,var,seed)
                print("Min DA: ", np.min(val))
                print("Max DA: ", np.max(val))
                print("Execution time: %s seconds" % (Exec_time))
                with open(filename, "a") as csvfile:
                    csvwriter =csv.writer(csvfile)
                    csvwriter.writerow(["DA",Exec_time,np.min(val),var,gr,seed])                

                #Method 3: DA_p
                start_time = time.time()
                val = [DigitalAnnealing2(pbf,pbf_var_dict,steps, Temp, i)[0] for i in range(1, Num_MC)]
                Exec_time = time.time() - start_time
                print(gr,var,seed)
                print("Min DA_p: ", np.min(val))
                print("Max DA_p: ", np.max(val))
                print("Execution time: %s seconds" % (Exec_time))
                with open(filename, "a") as csvfile:
                    csvwriter =csv.writer(csvfile)
                    csvwriter.writerow(["DA_p",Exec_time,np.min(val),var,gr,seed])    

def Track_Steps():
    seed=422423
    variables = 100
    degree = 2
    Temp = 100
    pbf = createPoly(variables,degree,0.6,seed=seed)
    steps=1500
    #pbf = {(1,): 1, (2,): -1}
    pbf_var_dict = createPolyDict(pbf,variables) 
    varAssignement = getInitialVarAssignement(pbf, seed)
    fig = make_subplots(rows=2,cols=1)
    if 1:
        start_time =time.time()
        val = DigitalAnnealing(pbf,pbf_var_dict, steps, Temp,initialvarAssignment=varAssignement,track_steps=True)

        #print(val2)
        print("Min Digital solution: ", np.min(val[0]))
        print("Max Digital solution: ", np.max(val[0]))
        print("Execution time normal DA: %s seconds" % (time.time() - start_time))
        csv = pd.DataFrame(np.array(val[2:]).transpose())
        fig.add_trace(go.Scatter(x=csv.index, y=csv[0],name="DA"),row=1,col=1)
        fig.add_trace(go.Scatter(x=csv.index, y=csv[1],name = "DA Ts"),row=2,col=1)
    if 1:
        start_time =time.time()
        val = DigitalAnnealing_experiment(pbf,pbf_var_dict, steps, Temp,initialvarAssignment=varAssignement,track_steps=True)

        #print(val2)
        print("Min Digital exp solution: ", np.min(val[0]))
        print("Max Digital exp solution: ", np.max(val[0]))
        print("Execution time normal DA: %s seconds" % (time.time() - start_time))
        csv = pd.DataFrame(np.array(val[2:]).transpose())
        fig.add_trace(go.Scatter(x=csv.index, y=csv[0],name="DA exp"),row=1,col=1)
        fig.add_trace(go.Scatter(x=csv.index, y=csv[1],name = "DA Ts exp"),row=2,col=1)        
    if 1:
        start_time =time.time()
        val = simulatedAnnealing(pbf,pbf_var_dict, steps, Temp,initialvarAssignment= varAssignement,track_steps=True)

        #print(val2)
        print("Min simlated solution: ", np.min(val[0]))
        print("Max simluated solution: ", np.max(val[0]))
        print("Execution time normal DA: %s seconds" % (time.time() - start_time))
        csv = pd.DataFrame(np.array(val[2]).transpose())
        fig.add_trace(go.Scatter(x=csv.index, y=csv[0],name = "SA"),row=1,col=1)

    if 1:
        start_time =time.time()
        val = SCA_Annealing(pbf,pbf_var_dict, steps, Temp,initialvarAssignment= varAssignement,track_steps=True)

        #print(val2)
        print("Min SCA solution: ", np.min(val[0]))
        print("Max SCA solution: ", np.max(val[0]))
        print("Execution time SA: %s seconds" % (time.time() - start_time))
        csv = pd.DataFrame(np.array(val[2:]).transpose())
        fig.add_trace(go.Scatter(x=csv.index, y=csv[0],name = "SCA"),row=1,col=1)
        #fig.add_trace(go.Scatter(x=csv.index, y=csv[1],name = "SCA Qs"))
        fig.add_trace(go.Scatter(x=csv.index, y=csv[2],name = "SCA Ts"),row=2,col=1)

    fig.update_layout(
        title_text="Hyperparameters: \n Vars = " +str(variables) + ", degree = " + str(degree) + ", steps = " +str(steps))
    fig.show()



#Run Script Shortcuts


if 1:
    Track_Steps()

if 0:
    testPerformance()
if 0:
    CompareEvaluations(1)
    CompareEvaluations(2)
    CompareEvaluations(3)
    CompareEvaluations(4)
