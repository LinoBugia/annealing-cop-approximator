import numpy as np
import pandas as pd
from p_tqdm import p_imap,t_map
import random
from p_tqdm import *
from itertools import permutations,combinations
import os
import csv
#from Funcs_Annealers import simulatedAnnealing
import plotly 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#evaluation functions



# EVALUATION FUNCTIONS


def Eval_Delta_Energy(pbf:dict[tuple:float],pbf_var_dict:dict[int:tuple], varAssignement:dict[int:bool],pos:int)-> float:

    """
    Input:  pbf:                a dictionary like function: {(1,2,4): 3.4, (3,): 7, ...} <-> 3.4 * x_1x_2x_4 + 7x_3
            pbf_var_dict:       a dicitonary for the pbf, which maps variables of occurences in the pbf
            varAssignement:     Assignement of variables as a dictionary: {1:0, 2:0, 3:1, 4:0, ...}
            pos:                Position of which the deltaE should be calculated

    returns: Evaluated delta E diffrence for position pos as a double
    """

    out = 0
    for monomial in pbf_var_dict:
        impacts = True
        for var in monomial:
            val = varAssignement[var]
            if val == 0 and var != pos:
                impacts = False
                break

        if impacts:
            out += pbf[monomial]
    if varAssignement[pos]==0:
        return out
    else: return -out

def Eval_Delta_Energy_flipped_var(pbf:dict[tuple:float],pbf_var_dict:dict[int:tuple], varAssignement:dict[int:bool],pos:int,pos2:int)-> float:

    """
    Input:  pbf:                a dictionary like function: {(1,2,4): 3.4, (3,): 7, ...} <-> 3.4 * x_1x_2x_4 + 7x_3
            pbf_var_dict:       a dicitonary for the pbf, which maps variables of occurences in the pbf
            varAssignement:     Assignement of variables as a dictionary: {1:0, 2:0, 3:1, 4:0, ...}
            pos:                Position of which the deltaE should be calculated

    returns: Evaluated delta E diffrence for position pos as a double
    """
    if varAssignement[pos2]==1:
        return 0
        # case vA[pos2] flipped = 0
    else: 
        out = 0
        for monomial in pbf_var_dict:
            impacts = True
            for var in monomial:
                val = varAssignement[var]
                if val == 0 and var != pos and var != pos2:
                    impacts = False
                    break

        if impacts:
            out += pbf[monomial]
        
        if varAssignement[pos]==0:    
            return out
        else:            
            return -out


def Eval_Delta_Energy_pos_pos2(pbf:dict[tuple:float],pbf_var_dict:dict[int:tuple], varAssignement:dict[int:bool],pos:int,pos2)-> float:
    #TODO UNSUED umbauen evntl.
    """
    Input:  pbf:                a dictionary like function: {(1,2,4): 3.4, (3,): 7, ...} <-> 3.4 * x_1x_2x_4 + 7x_3
            pbf_var_dict:       a dicitonary for the pbf, which maps variables of occurences in the pbf
            varAssignement:     Assignement of variables as a dictionary: {1:0, 2:0, 3:1, 4:0, ...}
            pos:                Position of which the deltaE should be calculated

    returns: Evaluated delta E diffrence for position pos as a double
    """

    out = 0
    for monomial in pbf_var_dict:
        impacts = True
        for var in monomial:
            val = varAssignement[var]
            if val == 0 and var != pos:
                impacts = False
                break

        if impacts:
            out += pbf[monomial]
    if varAssignement[pos]==0:
        return out
    else: return -out

def Eval_Delta_E_Mon_Bitflip(mon:int,x:int,var:int,pbf:dict[tuple:float], varAssignement:dict[int:bool]):
    
    #TODO UNUSED
    
    E = 0
    #E = -Eval_Delta_Energy({mon:pbf[mon]},[mon],varAssignement,x)     
    if varAssignement[var] == 0:varAssignement[var] = 1
    else:varAssignement[var] = 0    
    E2 = Eval_Delta_Energy({mon:pbf[mon]},[mon],varAssignement,x) 
    if varAssignement[var] == 0:
        E -= E2 
        varAssignement[var] = 1
    else:
        E += E2 
        varAssignement[var] = 0        
    return E

def evalPBF(pbf:dict[tuple:float], varAssignement:dict[int:bool]):
    """
    Input:  pbf: a dictionary like function: {(1,2,4): 3.4, (3,): 7, ...} <-> 3.4 * x_1x_2x_4 + 7x_3
            varAssignement: Assignement of variables as a dictionary: {1:0, 2:0, 3:1, 4:0, ...}

    returns: Evaluated pbf as a double
    """
    out=0
    for monomial in pbf:
        impacts = True
        for var in monomial:
            val = varAssignement[var]
            if val == 0:
                impacts = False
                break

        if impacts:
            out += pbf[monomial]
    return out
def evalPBF_List(pbf:dict[tuple:float], varAssignement:list):
    """
    Input:  pbf: a dictionary like function: {(1,2,4): 3.4, (3,): 7, ...} <-> 3.4 * x_1x_2x_4 + 7x_3
            varAssignement: Assignement of variables as a list: [0 ,0, 1, 0, ... , 0]

    returns: Evaluated pbf as a double
    """
    out=0
    for monomial in pbf:
        impacts = True
        for var in monomial:
            val = varAssignement[var]
            if val == 0:
                impacts = False
                break

        if impacts:
            out += pbf[monomial]
    return out
# modification (& evaluation) function

def modificate_varAssignement(varAssignement,var,method):

    """
    modifcates var_Assigment
    - method = 0: bitflip
    """


    if method == 0:
        if varAssignement[var] == 0:varAssignement[var] = 1
        else:varAssignement[var] = 0  
    return varAssignement

def modificate_update_deltaE(varAssignement,deltaE,vars:list,pbf,pbf_var_dict):

    """
    This function updates the deltaE vector for the DA algorithm (can also be used for the SCAs)
    """


    for var in vars:
        deltaE[var]=-deltaE[var]                                                                #korrekt
        for mon in pbf_var_dict[var]:                                                           #korrekt
            for x in mon:
                if x != var:

    ### TODO in eine Funktion umbauen mit params mon, pbf[mon], varAssignement, x , var 
                    #erase old contribution
                    E2 = Eval_Delta_Energy({mon:pbf[mon]},[mon],varAssignement,x)               #funktioniert!!!!
                    deltaE[x] -= E2 
                    #Test=Eval_Delta_Energy_flipped_var({mon:pbf[mon]},[mon],varAssignement,x,var)
                    if varAssignement[var] == 0:varAssignement[var] = 1
                    else:varAssignement[var] = 0    
                    E2 = Eval_Delta_Energy({mon:pbf[mon]},[mon],varAssignement,x) 
                    if varAssignement[var] == 0:
                        deltaE[x] -= E2 
                        varAssignement[var] = 1
                    else:
                        deltaE[x] += E2 
                        varAssignement[var] = 0     
        if varAssignement[var] == 0:varAssignement[var] = 1
        else:varAssignement[var] = 0  

    ### TODO in eine Funktion umbauen mit params mon, pbf[mon], varAssignement, x , var 

    return deltaE, varAssignement 

def update_deltaE_mon(varAssignement,mon,var,pbf):
    deltaE_List= []
    vars=[]
    for x in mon:
        if x != var:
            #erase old contribution
            E2 = -Eval_Delta_Energy({mon:pbf[mon]},[mon],varAssignement,x)               #funktioniert!!!!
            if varAssignement[var] == 0:
                deltaE_List.append(E2+Eval_Delta_Energy_flipped_var({mon:pbf[mon]},[mon],varAssignement,x,var) ) 
                vars.append(x)
            else:
                deltaE_List.append(-Eval_Delta_Energy_flipped_var({mon:pbf[mon]},[mon],varAssignement,x,var) + E2) 
                vars.append(x)
    return deltaE_List,vars

def eval_deltaE_many_bitflips(pbf:dict[tuple:float],pbf_var_dict:dict[int:tuple], varAssignement:dict[int:bool],pos_list:list[int]):
    
    """
    Evals the deltaE for a chain of bitflips
    """
    
    E=0
    for pos in pos_list:
        E += Eval_Delta_Energy(pbf,pbf_var_dict[pos], varAssignement,pos)
        varAssignement=modificate_varAssignement(varAssignement,pos,0)

    if 1:
        #flip back
        for pos in pos_list:
            varAssignement=modificate_varAssignement(varAssignement,pos,0)

    return E

# Landscape functions

def Print_VarAssigmentCommandline(input:list):
    print("---------")
    for k in range(int(np.sqrt(len(input)))):
        #print(str(k*np.sqrt(len(input))))
        print(input[k*int(np.sqrt(len(input))):(k+1)*int(np.sqrt(len(input)))])
    #1
    
def getPBFLandscape(pbf):
    """
    Input:  pbf: a dictionary like function: {(1,2,4): 3.4, (3,): 7, ...} <-> 3.4 * x_1x_2x_4 + 7x_3

    returns: Evaluated pbf at every possible value
    """
    varAssignement = getInitialVarAssignement(pbf)
    varList = []
    for var in varAssignement:
        varAssignement[var] = 0
        varList.append(var)
    
    return getPBFLandscapeHelper(pbf, varAssignement, varList, 0)

def getPBFLandscapeHelper(pbf, varAssignement, varList, varPos):
    if varPos >= len(varList):
        #print("varAssignement: ", varAssignement)
        val = evalPBF(pbf, varAssignement)
        #print("val: ", val)
        return [val]
    else:
        varAssignement[varList[varPos]] = 0
        left = getPBFLandscapeHelper(pbf, varAssignement, varList, varPos +1)
        varAssignement[varList[varPos]] = 1
        right = getPBFLandscapeHelper(pbf, varAssignement, varList, varPos +1)

        #print("left: ", left)
        #print("right: ", right)
        return left + right

# PBF MODIFICATION FUNCTIONS
    
def Add_entry_pbf(pbf,key,value):
    
    """
    This function adds a entry to the pbf, when the entry already exists it adds it to the existing
    """
    
    
    key=tuple(sorted(key))
    
    if pbf.__contains__(key):
        pbf[key] += value 
    else:
        pbf[key] = value 

    return pbf


def Sort_pbf(pbf,degree):
    
    """
    It redistributes double values pbf[(0,0)] -> pbf[(0,)]+=pbf[(0,0)])
    
    This function sorts the tuple of a pbf in order:
    (),
    (0,),(1,), (2,), (3,), ...
    (0,1), (0,2), ... , (1,1), (1,2) , ...
    ...
    (via degree then lexikographic in tuple)
    
    
    Input: pbf, degree
    
    Output: sorted pbf
    """
    
    
    
    new_pbf = dict()
    liste = list(pbf.keys())
    if () in liste:
        liste.remove(())
        new_pbf[()]=pbf[()]
    list_sorted = sorted(liste, key=lambda tup: tup[0:degree])



    for d in range(degree+1):
        for mon in list_sorted:
            if len(mon)==d and pbf[mon]!=0:
                new_pbf[mon]= pbf[mon]
                
    
    return new_pbf    

# algorithm functions

def getInitialVarAssignement(pbf:dict[tuple:float], seed=42)->dict[int:bool]:
    """
    generate a random initial variable assignement

    Input:  pbf: a dictionary like function: {(1,2,4): 3.4, (3,): 7, ...} <-> 3.4 * x_1x_2x_4 + 7x_3

    returns: Assignement of variables as a dictionary: {1:0, 2:0, 3:1, 4:0, ...}
    """
    random.seed(seed)
    varAssignement = dict()
    for monomial in pbf:
        for var in monomial:
            if not varAssignement.__contains__(var):
                varAssignement[var] = random.randint(0,1)
    
    #fill empty with zeros

    max = np.max(list(varAssignement.keys()))
    for i in range(max):
        if not varAssignement.__contains__(i):
                varAssignement[i] = random.randint(0,1)
                
    
    keys_sorted = sorted(varAssignement.keys(), key=lambda tup: tup)
    newvarAssignement=dict()
    for key in keys_sorted:
        newvarAssignement[key]=varAssignement[key]
    return newvarAssignement

def createPoly(variables: int, degree: int, density: float = 1.0, seed=42):
    """
    Creates polynomial dicts of the form {<monomial>: <alpha>} (e.g. {(1,2,3,4,5): 7.3})
    variables: number of variables in poly
    degree: polynomial's degree
    density: density for degree-k polynomials
    """
    random.seed(seed)
    out = dict()
    if variables < degree:
        raise ValueError("createPoly: degree must be at most #variables")
    varList = [x for x in range(0, variables)]

    for i in range(1, degree + 1):
        # create degree-i monomials:
        monomials = list(combinations(varList, i))
        for m in monomials:
            if random.random() < density:
                out[tuple(m)] = 1/2*random.random()*random.uniform(1,256) + random.random()
                if random.random() < random.random():
                    out[tuple(m)] = - out[tuple(m)]



    return out

def createPolyDict(pbf:dict[tuple:float],variables:int)-> dict[int:tuple]:

    """
    Creates a dictionary which maps a variable to its occurences in a monom in the pbf
    """

    out=dict()
    array = []
    for i in range(variables):
        array.append([])

    for mon in pbf:
        for var in mon:
            array[var].append(mon)

    for i in range(variables):
        out[i]=array[i]


    return out
    

def allow_anyway(delta:float, temperature:float,type=0)->bool:
    """
    Performs random experiment for simulated annealing, depending on the current temperature and the delta energy

    Returns True, when flip would be accepted and False otherwise

    type = 0 ---> Metropolis Kriterium (suited for SA,DA)
    type = 1 ---> Glauber Dynamics (suited for SCA)
    """
    if type == 0:
        if temperature <= 0:
            return False
        prob = np.min([np.exp((-delta)/temperature), 1])        #Boltzmann Distribution
    else:
        prob =  1/(1+np.exp((-delta)*temperature/2))              #Glauber Dynamics
    return False if random.random() > prob else True

def simulatedAnnealing_for_avg(pbf:dict[tuple:float],pbf_var_dict:dict[int:tuple],steps:int,
                       varAssignement:dict[int:bool],Ts:list[float],save_addinfo:bool,seed_rand:int)->list[list[float],list[bool]]:
    
    """
    Performs the Simulated Annealing Type Minimization Algorithm:

    Parameters:
    Input:  pbf:                a dictionary like function: {(1,2,4): 3.4, (3,): 7, ...} <-> 3.4 * x_1x_2x_4 + 7x_3
            pbf_var_dict:       a dicitonary for the pbf, which maps variables of occurences in the pbf
            varAssignement:     initial Assignement of variables as a dictionary: {1:0, 2:0, 3:1, 4:0, ...}
            steps:              number of steps
            Ts:                 list of a Cooling Schedule generated with Generate_Cooling_Schedule 
            save_addinfo:       bool if Min_VarAssignment should be saved
            seed_rand:          seed for random function

    Output: list of Energys, Min_VarAssignment 

    Main procedure of the algorithm: 

    each step:
        T = Ts[step] 
        select one variable uniformly
            Evaluate delta E
                accept if Delta_E < 0 or allowanyway(deltaE,T) == 1 (Metropolis Kriterium)

    """

    Min_varAssignement=[]
    random.seed(seed_rand)
    varsList = [k for k in varAssignement.keys()]
    vals = []
    E = evalPBF(pbf, varAssignement)
    vals.append(E)
    if save_addinfo: Min = E
    for step in range(steps):
        T = Ts[step]

        #Choose random variable for the next step
        var = varsList[random.randint(0, len(varsList)-1)]
        #switch variable

        deltaE = Eval_Delta_Energy(pbf, pbf_var_dict[var],varAssignement,var)
        #Trivially accept flip
        if deltaE > 0:
            if allow_anyway(deltaE, T,type = 0):
                if varAssignement[var] == 0:varAssignement[var] = 1
                else:varAssignement[var] = 0 
                E+=deltaE
        else: 
            E+=deltaE
            if varAssignement[var] == 0:varAssignement[var] = 1
            else:varAssignement[var] = 0 

        vals.append(E)           
        if save_addinfo:
            if E <= Min:
                Min = E
                Min_varAssignement =list(varAssignement.values())
        #test if eval correct
        #if step == 0: vals2=[]
        #vals2.append(evalPBF(pbf,varAssignement))
    #print(evalPBF(pbf,Min_varAssignement))
    return vals,Min_varAssignement
def Generate_Cooling_Schedule(cooling_param:list,steps:int)->list[float]:
    """
    generates a cooling schedule from the cooling_param as a list
    """
    T=[]
    for step in range(1,steps+1):
        if cooling_param[0] ==  "constant":
            C =cooling_param[1]
            T.append(C) 
        if cooling_param[0] ==  "linear":
            T_start =cooling_param[1]
            T_end = cooling_param[2]
            T.append(T_start + step*(T_end-T_start)/steps) 
        elif cooling_param[0] ==  "stepfunction":
            T_start =cooling_param[1]
            T_end = cooling_param[2]
            Number_of_Steps = cooling_param[3]
            T.append()
        elif cooling_param[0] ==  "linear&logarithmic":
            T_start =cooling_param[1]
            T_end = cooling_param[2]
            Number_of_Steps = cooling_param[3]
            T.append()
        elif cooling_param[0] ==  "rising":
            T_start =cooling_param[1]
            T.append(T_start*step)
        elif cooling_param[0] ==  "logarithmic":
            c = cooling_param[1]
            T.append(c/(np.log(1+pow(step, 2.22))))
            
        elif cooling_param[0] ==  "logarithmic_step":
            c = cooling_param[1]
            Number_of_Steps = cooling_param[2]
            for _ in range(Number_of_Steps):
                T.append((np.log(1+step))/c)
            
        elif cooling_param[0] ==  "exponential":
            c =cooling_param[1]
            T.append(np.exp(step/c)-1)
            
            
        elif cooling_param[0] == "auto":
            steps_init = cooling_param[1]
            steps_avg = cooling_param[2]
            delta = cooling_param[3]
            E = cooling_param[3]
            pbf = cooling_param[5]
            pbf_var_dict = cooling_param[6]
            initial_varAssignement = cooling_param[7]
            varAssignement = initial_varAssignement.copy()
            varsList = [k for k in varAssignement.keys()]
            
            c = 0
            m_1 =0
            m_2 = 0
            delta_f_plus = 0
            deltaEs=[] 
            cs=[]
            for step in range(steps_init):
                #print(str(np.log(m_2/abs(m_2*p -m_1*(1-p)))))
                if step>0:
                    if c ==0: p= 0.99
                    else: p= np.min([np.exp((-deltaE)/c), 1])/1.1
                    if m_1>0:
                        c = delta_f_plus/(np.log(m_2/abs(m_2*p -m_1*(1-p))))
                        if c==float("inf") or c==float("nan"):c=0
                        else: 
                            if c > 0: cs.append(c)
                        if p==float("nan"): p=0
                #if c == float("inf"): c=10000
                #Choose random variable for the next step
                var = varsList[random.randint(0, len(varsList)-1)]
                    #switch variable

                deltaE = Eval_Delta_Energy(pbf, pbf_var_dict[var],varAssignement,var)
                #Trivially accept flip
                if deltaE > 0:
                    m_2 +=1
                    if allow_anyway(deltaE, c,type = 0):
                        if varAssignement[var] == 0:varAssignement[var] = 1 #mit oder ohne flip?
                        else:varAssignement[var] = 0 
                else: 
                    if varAssignement[var] == 0:varAssignement[var] = 1 #mit oder ohne flip?
                    else:varAssignement[var] = 0 
                    m_1 += 1
                if deltaE >0:
                    deltaEs.append(deltaE)
                    delta_f_plus=np.array(deltaEs).mean()
            c = np.array(cs).mean()
            T.append(c)
            for step in range(steps):
                
                # calc mean and standard deviation from MC trials
                Results = simulatedAnnealing_for_avg(pbf,pbf_var_dict,steps_avg,initial_varAssignement.copy(),[T[step]]*steps_avg,save_addinfo=False,seed_rand=random.uniform(0,100))
                # Mean
                std_deviation = np.std(np.array(Results[0]))
                alpha = np.log(1+delta)/(1.23*std_deviation)
                    
                T.append(T[step]/(1+alpha*T[step]))
            break
    return T[:steps]

def Generate_Number_Partitioning_List(length,upper_bound):
    return [random.randint(0, upper_bound) for p in range(0, length)] 


# PLOTTING FUNCTIONS

def Plot_Trajec_nice(Trajectories):
    """
    This function plots two list of trajecotires in the same plot

    """
    colors =[]
    fig = make_subplots(rows=1,cols=1
                        )
    if 0:
        for i in range(100):
            k = np.mod(i,23)
            colors.append(plotly.colors.qualitative.Dark24_r[k])
    else:
        colors=["blue","red"]
    for i in range(len(Trajectories)):
        for j in range(len(Trajectories[i])):
            csv2 = pd.DataFrame(np.array(Trajectories[i][j]).transpose())
            fig.add_trace(go.Scatter(x=csv2.index, y=csv2[0],name="MC"+str(i),legendgroup="MC"+str(i),marker=dict(color=colors[i])),row=1,col=1) 
    
    
    if 0:
        fig.update_layout(
        xaxis=dict(
            showgrid=False,
            zeroline=True,
            showticklabels=False,
            ticks='',
            title=None
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            ticks='',
            title=None
        ),
        showlegend=False,       # Hides the legend
        plot_bgcolor='white',   # Removes gray background
        paper_bgcolor='white',  # Removes outer background
        margin=dict(l=0, r=0, t=0, b=0)  # Optional: remove all margins
        ) 
    fig.show()

    
    
def Save_Results_As_List(problem, type_alg, name, pbf_size, total_time, steps, cooling_parm, Mins):
    """
    This function saves a row in a csv file
    """
    
    file_path = "/Users/lino/Documents/python/fujitsuclone/Code/data_thesis.txt"
    with open(file_path, "a") as csvfile:
        csvwriter =csv.writer(csvfile)
        row = [problem, type_alg, name, pbf_size, total_time, steps, cooling_parm, str(Mins)] #TODO
        csvwriter.writerow(row)    
    
    1
# HELPER FUNCTIONS GTP
def generate_one_line_one_rest_zero_Matrix(r,s,loc):
    
    """
    Creates Matrix needed for GTP Problem
    
    """
    
    M=[]
    for i in range(r):
        if i == loc:
            arr = np.ones(s)
        else:
            arr = np.zeros(s)
        if i > 0:
            M=np.vstack((M,arr))
        else:
            if loc==0:
                M = np.ones(s)
            else: M = np.zeros(s)
    return M

def Generate_GTP(total_number, r,s, seed_rand):
    """
    Generates a matrix out of the parameters
    """
    random.seed(seed_rand)
    
    r_h = total_number
    s_h = total_number
    
    b =[]
    for i in range(r-1):
        val = round(random.uniform(3,r_h/(0.55*r)))
        b.append(val)
        r_h -= val
    b.append(r_h)    
    for i in range(s-1):
        val = round(random.uniform(3,s_h/(0.55*s)))
        b.append(val)
        s_h -= val
    b.append(s_h)    
    # random transport costs
    c = []
    for _ in range(r):
        for _ in range(s):
            c.append(random.uniform(500,1500))
            
    # generate the matrix
    A_1=generate_one_line_one_rest_zero_Matrix(r,s,0)
    A_2=np.identity(s)
    for i in range(1,r):
        M_1 = generate_one_line_one_rest_zero_Matrix(r,s,i)
        A_1 = np.hstack((A_1,M_1))
        M_2 = np.identity(s)
        A_2 = np.hstack((M_2,A_2))
        
    A= np.vstack((A_1,A_2))
    
    return A,np.array(b),np.array(c)