import ctypes
import multiprocessing 
from Funcs_Annealing2 import *
from multiprocessing import Value,Array
import os
import time


def simulatedAnnealing(pbf:dict[tuple:float],pbf_var_dict:dict[int:tuple],steps:int,
                       varAssignement:dict[int:bool],Ts:list[float],save_addinfo:bool,seed_rand:int,Ising=False)->list[list[float],list[bool]]:
    
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
    result_List=[]
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
        if Ising: result_List.append(list(varAssignement.values()))
        #test if eval correct
        #if step == 0: vals2=[]
        #vals2.append(evalPBF(pbf,varAssignement))
    #print(evalPBF(pbf,Min_varAssignement))
    return vals,Min_varAssignement,result_List

        
      
def SCA_Annealing(pbf:dict[tuple:float],pbf_var_dict:dict[int:tuple],steps:int,
                       varAssignement:dict[int:bool],Ts:list[float],Qs:list[float],save_addinfo:bool,seed_rand:int)->list[list[float],list[bool]]:

    """
    Performs the SCA Type Minimization Algorithm:

    Parameters:
    Input:  pbf:                a dictionary like function: {(1,2,4): 3.4, (3,): 7, ...} <-> 3.4 * x_1x_2x_4 + 7x_3
            pbf_var_dict:       a dicitonary for the pbf, which maps variables of occurences in the pbf
            varAssignement:     initial Assignement of variables as a dictionary: {1:0, 2:0, 3:1, 4:0, ...}
            steps:              number of steps
            Ts:                 list of a Cooling Schedule generated with Generate_Cooling_Schedule 
            Qs:                 list of Qs generated with Generate_Cooling_Schedule
            save_addinfo:       bool if Min_VarAssignment should be saved
            seed_rand:          seed for random function

    Output: list of Energys, Min_VarAssignment 

    Main procedure of the algorithm: 

    each step:
        T = Ts[step] 
        Q = Qs[step]
            For each variable:
                Evaluate delta E
                    allowanyway(deltaE + Q ,T) == 1 (GlauberDynamics)
            Flip ALL selected 

    """
    Min_varAssignement=[]
    random.seed(seed_rand)
    vals = []
    E=evalPBF(pbf, varAssignement)
    vals.append(evalPBF(pbf, varAssignement))   
    if save_addinfo: Min = E
    for step in range(steps):
        T = Ts[step]
        Q = Qs[step]
        select=[]
        deltaE=[0]*len(varAssignement)
        for i in range(len(varAssignement)):
            deltaE[i]=Eval_Delta_Energy(pbf, pbf_var_dict[i],varAssignement,i)
            if allow_anyway(deltaE[i]+Q,T,type =1):
                select.append(i)
        for var in select:
            #variable switch
            if varAssignement[var] == 0: 
                varAssignement[var] = 1 
            else: 
                varAssignement[var] = 0   
            E += deltaE[var]

        vals.append(E)
        if save_addinfo:
            if E <= Min:
                Min = E
                Min_varAssignement =list(varAssignement.values())
        #test if eval correct
        if step == 0: vals2=[]
        vals2.append(evalPBF(pbf,varAssignement))
    return vals2,Min_varAssignement


def DigitalAnnealing_old(E_init:float,delta_E_init,pbf:dict[tuple:float],pbf_var_dict:dict[int:tuple],steps:int,varAssignement:dict[int:bool],
                     Ts:list[float],save_addinfo:bool,seed_rand:int,offset_increase_rate:int)->list[list[float],list[bool]]:

    """
    Performs the Digital Annealing Type Minimization Algorithm:

    Parameters:
    Input:  pbf:                    a dictionary like function: {(1,2,4): 3.4, (3,): 7, ...} <-> 3.4 * x_1x_2x_4 + 7x_3
            pbf_var_dict:           a dicitonary for the pbf, which maps variables of occurences in the pbf
            varAssignement:         initial Assignement of variables as a dictionary: {1:0, 2:0, 3:1, 4:0, ...}
            steps:                  number of steps
            Ts:                     list of a Cooling Schedule generated with Generate_Cooling_Schedule 
            save_addinfo:           bool if Min_VarAssignment should be saved
            seed_rand:              seed for random function
            offset_increase_rate:   int of increase of E_offset if no variable is selected for flip

    Output: list of Energys, Min_VarAssignment 

    Main procedure of the algorithm: 

    E_Offset = 0
    each step:
        T = Ts[step] 
            For each variable:
                Evaluate delta E
                    allowanyway(delta_E + E_Offset,T) == 1 (Metropolis Kriterium) --> add to select
            Select uniform out of select 
                flip selected 
                E_Offset = 0
            if select = []
                E_Offset = E_Offset + offset_increase_rate
    """


    #TODO offset dynamisch bestimmen?
    E_offset=0
    Min_varAssignement=[]
    random.seed(seed_rand)
    varsList = [k for k in varAssignement.keys()]
    vals = []
    #deltaE=delta_E_init
    E = E_init  
    vals.append(E)
    if save_addinfo: Min = E
    deltaE=[0]*len(varAssignement)
    for step in range(0,steps):
        T = Ts[step]
        #compute deltaE vektor & list for select
        #deltaE=[0]*len(varAssignement)
        select = []
        # Calculate Metropolis Kriterion
        for i in varsList:
            deltaE[i]=Eval_Delta_Energy(pbf, pbf_var_dict[i],varAssignement,i)
            if deltaE[i]>0:
                if allow_anyway(deltaE[i]-E_offset,T):
                    select.append(i)
            else:
                select.append(i)
        #Choose random variable out of the "zulässigen" for the variable switch
        if select != []: 
            var = select[random.randint(0, len(select)-1)]
            #variable switch
            if varAssignement[var] == 0:varAssignement[var] = 1
            else:varAssignement[var] = 0          
            E_offset=0
            E+=deltaE[var]


        else:
            E_offset += offset_increase_rate 
        vals.append(E)           
        if save_addinfo:
            if E <= Min:
                Min = E
                Min_varAssignement =list(varAssignement.values())
                
        #test if eval correct
        #if step == 0: vals2=[]
        #vals2.append(evalPBF(pbf,varAssignement))
    print(evalPBF(pbf,Min_varAssignement))
    return vals,Min_varAssignement


def DigitalAnnealing(E_init:float,deltaE:list[float],pbf:dict[tuple:float],pbf_var_dict:dict[int:tuple],steps:int,varAssignement:dict[int:bool],
                     Ts:list[float],save_addinfo:bool,seed_rand:int,offset_increase_rate:int,Ising=False)->list[list[float],list[bool]]:

    """
    Performs the Digital Annealing Type Minimization Algorithm:

    Parameters:
    Input:  pbf:                    a dictionary like function: {(1,2,4): 3.4, (3,): 7, ...} <-> 3.4 * x_1x_2x_4 + 7x_3
            pbf_var_dict:           a dicitonary for the pbf, which maps variables of occurences in the pbf
            varAssignement:         initial Assignement of variables as a dictionary: {1:0, 2:0, 3:1, 4:0, ...}
            steps:                  number of steps
            Ts:                     list of a Cooling Schedule generated with Generate_Cooling_Schedule 
            save_addinfo:           bool if Min_VarAssignment should be saved
            seed_rand:              seed for random function
            offset_increase_rate:   int of increase of E_offset if no variable is selected for flip

    Output: list of Energys, Min_VarAssignment 

    Main procedure of the algorithm: 

    E_Offset = 0
    each step:
        T = Ts[step] 
            For each variable:
                Evaluate delta E
                    allowanyway(delta_E + E_Offset,T) == 1 (Metropolis Kriterium) --> add to select
            Select uniform out of select 
                flip selected 
                E_Offset = 0
            if select = []
                E_Offset = E_Offset + offset_increase_rate
    """

    result_List=[]
    #TODO offset dynamisch bestimmen?
    E_offset=0
    Min_varAssignement=[]
    random.seed(seed_rand)
    varsList = [k for k in varAssignement.keys()]
    vals = []
    E = E_init  
    vals.append(E)
    if save_addinfo: Min = E
        #deltaE2=deltaE.copy()
    for step in range(0,steps):
        T = Ts[step]
        # Calculate Metropolis Kriterion
        select = []
        for i in varsList:
            if deltaE[i]>0:
                if allow_anyway(deltaE[i]-E_offset,T):
                    select.append(i)
            else:
                select.append(i)
        #Choose random variable out of the "zulässigen" for the variable switch
        if select != []: 
            var = select[random.randint(0, len(select)-1)]      
            E_offset=0
            E+=deltaE[var]
            deltaE, varAssignement =modificate_update_deltaE(varAssignement,deltaE,[var],pbf,pbf_var_dict)
  
        else:
            #E_offset += np.max(deltaE[var])*10000#np.mean(deltaE)
            E_offset += offset_increase_rate 
        vals.append(E)      
  
        if save_addinfo:
            if E <= Min:
                Min = E
                Min_varAssignement = list(varAssignement.values())
        if Ising: result_List.append(list(varAssignement.values()))
        #print(E,evalPBF(pbf,varAssignement))
        #test if eval correct
        #if step == 0: vals2=[]
        #vals2.append(evalPBF(pbf,varAssignement))
    #print(evalPBF_List(pbf,Min_varAssignement))
    return vals,Min_varAssignement,result_List

def DA_Step(input_args):
    mon,var,varAssignement = input_args
    #deltaE=[0]*len(list(varAssignement))
    # flip
    deltaE_List,vars = update_deltaE_mon(varAssignement,mon,var,pbf_p)    

    return deltaE_List,vars

def Random_Step(input_args):
    deltaE,T,var = input_args
    if allow_anyway(deltaE,T):
        return var

def pool_initializer(pbf):
    global pbf_p
    pbf_p = pbf
    
    

def DigitalAnnealing_parallel(pool,E_init:float,deltaE:list[float],pbf,pbf_var_dict:dict[int:tuple],steps:int,varAssignement:dict[int:bool],
                     Ts:list[float],save_addinfo:bool,seed_rand:int,offset_increase_rate:int)->list[list[float],list[bool]]:


    """
    Performs the Digital Annealing Type Minimization Algorithm:

    Parameters:
    Input:  pbf:                    a dictionary like function: {(1,2,4): 3.4, (3,): 7, ...} <-> 3.4 * x_1x_2x_4 + 7x_3
            pbf_var_dict:           a dicitonary for the pbf, which maps variables of occurences in the pbf
            varAssignement:         initial Assignement of variables as a dictionary: {1:0, 2:0, 3:1, 4:0, ...}
            steps:                  number of steps
            Ts:                     list of a Cooling Schedule generated with Generate_Cooling_Schedule
            save_addinfo:           bool if Min_VarAssignment should be saved
            seed_rand:              seed for random function
            offset_increase_rate:   int of increase of E_offset if no variable is selected for flip

    Output: list of Energys, Min_VarAssignment 

    Main procedure of the algorithm: 

    E_Offset = 0
    each step:
        T = Ts[step] 
            For each variable in parallel:
                Evaluate delta E
                    allowanyway(delta_E + E_Offset,T) == 1 (Metropolis Kriterium) --> add to select
            Select uniform out of select 
                flip selected 
                E_Offset = 0
            if select = []
                E_Offset = E_Offset + offset_increase_rate
    """
    

    #TODO offset dynamisch bestimmen?
    E_offset=0
    Min_varAssignement=[]
    random.seed(seed_rand)
    vals = []
    E = E_init
    vals.append(E)
    if save_addinfo: Min = E
    total_cores = 9
    cores_to_use = max(1, total_cores - 1)
    
    vars = list(range(len(varAssignement.keys()) ))
    k,m = divmod(len(vars),cores_to_use)
    #batches = [vars[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(total_cores)]
    #print(current_memory_usage())
    #pbf = Array(ctypes.py_object,pbf)
    #with multiprocessing.Pool(initializer=init,processes=cores_to_use,initargs=(pbf)) as pool:

    if 1:
    #args = (pbf,pbf_var_dict)
    #with multiprocessing.Pool(processes=cores_to_use,initializer=pool_initializer,initargs=args) as pool:
        #print(current_memory_usage())
        for step in range(0,steps):
            time_stamp = time.time()
            T = Ts[step]
            #paralell part
            #output = t_map(lambda x: DA_Step(pbf, pbf_var_dict[x], varAssignement,T,E_offset,x), eval,disable=True)
                # Calculate Metropolis Kriterion
            select = []
            #for i in vars:

            #    else:
            #        select.append(i)              

            if 0:
                input_args1=[(deltaE[var]+E_offset,T,var)for var in vars]
                select = pool.map_async(func = Random_Step,iterable=input_args1,chunksize  =k)
            else:
                for i in vars:
                    if deltaE[i]>0:
                        if allow_anyway(deltaE[i]-E_offset,T):
                            select.append(i)                
            #input_args = [(T,E_offset,batch,pbf.copy(),pbf_var_dict,varAssignement.copy()) for batch in batches]
                #print(current_memory_usage())
                #create batches
                #for i in range(len(deltaE)):
                #    if Out[i]:
                #        select.append(i)
            # Use starmap to pass multiple arguments
            #print("finished")
            #with multiprocessing.Pool() as pool:
            #    parallel = pool.imap(lambda x: DA_Step(pbf, varAssignement,T,x), eval)
            print(time.time()-time_stamp)
            time_stamp = time.time()
            
            if select != []: 
                var = select[random.randint(0, len(select)-1)]
                #variable switch
                mon_list = pbf_var_dict[var]     
                E_offset=0
                E+=deltaE[var]
                deltaE[var]=-deltaE[var]     
                input_args2 = [(mon,var,varAssignement) for mon in mon_list]
                Out = pool.map(func = DA_Step,iterable=input_args2,chunksize  =k)
                print(time.time()-time_stamp)
                time_stamp = time.time()
                
                for i in range(len(Out)):
                    for j in range(len(Out[i][0])):
                        deltaE[Out[i][1][j]] += Out[i][0][j]
                # variable switch
                if varAssignement[var] == 0:varAssignement[var] = 1
                else:varAssignement[var] = 0 
                  
            else:
                E_offset += offset_increase_rate 
            print(time.time()-time_stamp)
            time_stamp = time.time()
            print("___")
            vals.append(E)           
            if save_addinfo:
                if E <= Min:
                    Min = E
                    Min_varAssignement =list(varAssignement.values())
        #test if eval correct
        #if step == 0: vals2=[]
        print(evalPBF_List(pbf,Min_varAssignement))
        #vals2.append(evalPBF(pbf,varAssignement))
    return vals,Min_varAssignement



def DigitalAnnealing_experiment(pbf, pbf_var_dict,steps,initialTemperature, initialvarAssignment=[], track_steps=0, seed=42):
    # TODO UNFERTIG für neue Bib
    offset_increase_rate = 0   #TODO offset dynamisch bestimmen?
    if initialvarAssignment==[]:
        varAssignement = getInitialVarAssignement(pbf, seed)

    else:
        varAssignement=initialvarAssignment.copy()
    random.seed(seed)
    E_offset = 100
    vals = []
    Ts = []
    if track_steps:
        vals.append(evalPBF(pbf, varAssignement))   
        Ts.append(initialTemperature) 
    for step in range(1, steps +1):
        T = step * initialTemperature
        deltaE=[0]*len(varAssignement)
        for i in range(len(varAssignement)):
            deltaE[i]=Eval_Delta_Energy(pbf, pbf_var_dict[i],varAssignement,i)
        var = -1
        E_Probs = []
        #Version 1
        if 0:
            for i in range(len(deltaE)):
                E_Probs.append(float(np.min([np.exp((-deltaE[i])/T), 1])))
                #E_Probs.append(1)
        else:
            # shifte alle Werte ins positive:
            E_Probs = deltaE + np.min(deltaE)
        
        # Transformation zu Wahrscheinlichkeitsmaß
        length = float(np.sum(E_Probs))
        E_Probs = (1/length)*np.array(E_Probs)
        #print(np.sum(E_Probs))
        rand = random.random()
        #make a cummulated vector
        E=0
        for i in range(len(E_Probs)):
            E=E_Probs[i]+E
            if E > rand:
                var = i
                break
        #Choose random variable out of the "zulässigen" for the variable switch

        
            if 1:
                if varAssignement[var] == 0:
                    varAssignement[var] = 1
                else:
                    varAssignement[var] = 0
            E_offset=0


        if track_steps:
            vals.append(evalPBF(pbf, varAssignement))
            Ts.append(1/T)
    former_energy=evalPBF(pbf, varAssignement)

    return former_energy, varAssignement,vals,Ts


#Annealers for TSP:

def digitalAnnealing_TSP(E_init,deltaE_init,pbf,pbf_var_dict,steps,varAssignement,Ts,seed_rand,inv_trans_dict,offset_increase_rate ,save_addinfo=1)->list[list[float],list[bool]]:
    random.seed(seed_rand)
    E_Offset = 0
    vals = []
    E = E_init
    deltaE = deltaE_init
    Min = E
    Min_varAssignement = list(varAssignement.values())
    vals.append(E)
    cities = int(np.sqrt(len(varAssignement.keys())))
    combs = list(combinations(range(cities),2))




    for step in range(steps):
        select = []
        T = Ts[step]
        
        # variable flip check
        for var in range(len(combs)):
            if deltaE[var] > 0:
                if allow_anyway(deltaE[var]-E_Offset,T):  
                    select.append(var)
            else:
                select.append(var)

        
        if select != []:
            s = select[random.randint(0, len(select)-1)]
            # accept flip and update E
            #search right cities for flip
            var_1_update = combs[s][0]
            var_2_update = combs[s][1]
            E += deltaE[s]
            
            
            deltaE[s] = -deltaE[s]
            #E2 = evalPBF(pbf,varAssignement)
            #switch var Ass

            for i in range(cities):
                if varAssignement[inv_trans_dict[var_1_update,i]] == 1:
                    var_1_0 = inv_trans_dict[var_1_update,i]
                    break
            for j in range(cities):
                if varAssignement[inv_trans_dict[var_2_update,j]] == 1:
                    var_2_0 = inv_trans_dict[var_2_update,j]
                    break  

            var_2_1 = inv_trans_dict[var_2_update,i]
            var_1_1 = inv_trans_dict[var_1_update,j]

            varAssignement[var_1_0] = 0
            varAssignement[var_2_0] = 0 
            varAssignement[var_2_1] = 1 
            varAssignement[var_1_1] = 1   
            #E2 = evalPBF(pbf,varAssignement)
            if combs[s][0] == 0 or combs[s][1] == 0 or combs[s][0]==cities-1 or combs[s][1] == cities-1:              #var_1_update ==cities or var_2_update == cities*cities or var_1_update == 0 or var_2_update == 0 :
                switch_list= list(set([combs[s][0],combs[s][1],int(np.mod(combs[s][0]+1,cities)),int(np.mod(combs[s][1]-1,cities)),int(np.mod(combs[s][1]+1,cities)),int(np.mod(combs[s][0]-1,cities))]))
            else:
                switch_list = list(set([combs[s][0],combs[s][1],combs[s][0]+1,combs[s][1]-1,combs[s][0]-1,combs[s][1]+1]))
            for x in range(len(combs)): #update variable vector
                # only update affected E_Delta by the switch?
                if x != s:
                    if combs[x][0] in switch_list or combs[x][1] in switch_list:
                        var_1 = combs[x][0]
                        var_2 = combs[x][1]                                          #search right cities
                        for i in range(cities):
                            if varAssignement[inv_trans_dict[var_1,i]] == 1:
                                var_1_0 = inv_trans_dict[var_1,i]
                                break
                        for j in range(cities):
                            if varAssignement[inv_trans_dict[var_2,j]] == 1:
                                var_2_0 = inv_trans_dict[var_2,j]
                                break  
                        var_2_1 = inv_trans_dict[var_2,i]
                        var_1_1 = inv_trans_dict[var_1,j]
                            # add new contribution
                        
                        xv = eval_deltaE_many_bitflips(pbf,pbf_var_dict, varAssignement,[var_1_0,var_2_0,var_1_1,var_2_1])
                        deltaE[x] = xv
            
            #switch _var Ass

            E_Offset=0
            #print(s)
        else:
            E_Offset += random.uniform(1,offset_increase_rate)
        vals.append(E) 

        #print(E)
        #print(evalPBF(pbf,varAssignement))
        #print(combs[s])
        #if abs(E-evalPBF(pbf,varAssignement))>0.03:
        #    print(combs[s])                    #TSP RUNDUNGSFEHLER????
            
        if save_addinfo:
            if E < Min:
                Min = E
                Min_varAssignement =list(varAssignement.values())

        #test if eval correct
        #if step == 0: vals2=[]
        #vals2.append(evalPBF(pbf,varAssignement))                       
    #print(evalPBF(pbf,Min_varAssignement))
    return vals,Min_varAssignement


#Annealers for TSP:

def DA_Step_TSP(input_args):
    mon,var,varAssignement = input_args
    deltaE=[0]*len(list(varAssignement))
    # flip
    
    deltaE =update_deltaE_mon(varAssignement,deltaE,mon,var,pbf_p)    

    return [deltaE]

def pool_initializer_TSP(pbf,pbf_var_dict):
    global pbf_p
    pbf_p = pbf
    
def digitalAnnealing_TSP(E_init,deltaE_init,pbf,pbf_var_dict,steps,varAssignement,Ts,seed_rand,inv_trans_dict,offset_increase_rate ,save_addinfo=1)->list[list[float],list[bool]]:
    random.seed(seed_rand)
    E_Offset = 0
    vals = []
    E = E_init
    deltaE = deltaE_init
    Min = E
    Min_varAssignement = list(varAssignement.values())
    vals.append(E)
    cities = int(np.sqrt(len(varAssignement.keys())))
    combs = list(combinations(range(cities),2))




    for step in range(steps):
        select = []
        T = Ts[step]
        
        # variable flip check
        for var in range(len(combs)):
            if deltaE[var] > 0:
                if allow_anyway(deltaE[var]-E_Offset,T):  
                    select.append(var)
            else:
                select.append(var)

        
        if select != []:
            s = select[random.randint(0, len(select)-1)]
            # accept flip and update E
            #search right cities for flip
            var_1_update = combs[s][0]
            var_2_update = combs[s][1]
            E += deltaE[s]
            
            
            deltaE[s] = -deltaE[s]
            #E2 = evalPBF(pbf,varAssignement)
            #switch var Ass

            for i in range(cities):
                if varAssignement[inv_trans_dict[var_1_update,i]] == 1:
                    var_1_0 = inv_trans_dict[var_1_update,i]
                    break
            for j in range(cities):
                if varAssignement[inv_trans_dict[var_2_update,j]] == 1:
                    var_2_0 = inv_trans_dict[var_2_update,j]
                    break  

            var_2_1 = inv_trans_dict[var_2_update,i]
            var_1_1 = inv_trans_dict[var_1_update,j]

            varAssignement[var_1_0] = 0
            varAssignement[var_2_0] = 0 
            varAssignement[var_2_1] = 1 
            varAssignement[var_1_1] = 1   
            #E2 = evalPBF(pbf,varAssignement)
            if combs[s][0] == 0 or combs[s][1] == 0 or combs[s][0]==cities-1 or combs[s][1] == cities-1:              #var_1_update ==cities or var_2_update == cities*cities or var_1_update == 0 or var_2_update == 0 :
                switch_list= list(set([combs[s][0],combs[s][1],int(np.mod(combs[s][0]+1,cities)),int(np.mod(combs[s][1]-1,cities)),int(np.mod(combs[s][1]+1,cities)),int(np.mod(combs[s][0]-1,cities))]))
            else:
                switch_list = list(set([combs[s][0],combs[s][1],combs[s][0]+1,combs[s][1]-1,combs[s][0]-1,combs[s][1]+1]))
            
            
            
            for x in range(len(combs)):
                #update variable vector
                # only update affected E_Delta by the switch?
                if x != s:
                    if combs[x][0] in switch_list or combs[x][1] in switch_list:
                        var_1 = combs[x][0]
                        var_2 = combs[x][1]                                          #search right cities
                        for i in range(cities):
                            if varAssignement[inv_trans_dict[var_1,i]] == 1:
                                var_1_0 = inv_trans_dict[var_1,i]
                                break
                        for j in range(cities):
                            if varAssignement[inv_trans_dict[var_2,j]] == 1:
                                var_2_0 = inv_trans_dict[var_2,j]
                                break  
                        var_2_1 = inv_trans_dict[var_2,i]
                        var_1_1 = inv_trans_dict[var_1,j]
                            # add new contribution
                        
                        xv = eval_deltaE_many_bitflips(pbf,pbf_var_dict, varAssignement,[var_1_0,var_2_0,var_1_1,var_2_1])
                        deltaE[x] = xv
            
            #switch _var Ass

            E_Offset=0
            #print(s)
        else:
            E_Offset += random.uniform(1,offset_increase_rate)
        vals.append(E) 

        #print(E)
        #print(evalPBF(pbf,varAssignement))
        #print(combs[s])
        #if abs(E-evalPBF(pbf,varAssignement))>0.03:
        #    print(combs[s])                    #TSP RUNDUNGSFEHLER????
            
        if save_addinfo:
            if E < Min:
                Min = E
                Min_varAssignement =list(varAssignement.values())

        #test if eval correct
        #if step == 0: vals2=[]
        #vals2.append(evalPBF(pbf,varAssignement))                       
    #print(evalPBF(pbf,Min_varAssignement))
    return vals,Min_varAssignement


def simulatedAnnealing_TSP(pbf, pbf_var_dict, inv_trans_dict, cities, steps, initialTemperature, initialvarAssignment=[], track_steps=0, seed=42):
    
    # TODO UNFERTIG für neue Bib

    if initialvarAssignment==[]:
        varAssignement = getInitialVarAssignement(pbf, seed)
    else:
        varAssignement=initialvarAssignment.copy()
    random.seed(seed)
    varsList = [k for k in varAssignement.keys()]
    K=7500
    Ts=[]   
    vars = []

    Min_varAssignement = initialvarAssignment
    vals = []
    Energy = evalPBF(pbf, varAssignement)
    Minimum = Energy
    if track_steps:
        vals.append(Energy)
        Minimum = vals[0] 
    T = initialTemperature 
    a=1
    b= 1
    combs = list(combinations(range(cities),2))
    for step in range(1, steps +1):
        if np.mod(step,K)==0:
        #T = a/(b*np.log(step * initialTemperature))
            T = T + initialTemperature
            K = K-100
        #Choose 2 random cities for the switch

        select = varsList[random.randint(0, len(combs)-1)]
        var_1 = combs[select][0]
        var_2 = combs[select][1]

        for i in range(cities):
            if varAssignement[inv_trans_dict[var_1,i]] == 1:
                var_1_0 = inv_trans_dict[var_1,i]
                #deltaE2 = Eval_Delta_Energy(pbf, pbf_var_dict[var_1_c],varAssignement,var_1_c)
                break
        for j in range(cities):
            if varAssignement[inv_trans_dict[var_2,j]] == 1:
                var_2_0 = inv_trans_dict[var_2,j]
                #deltaE2 += Eval_Delta_Energy(pbf, pbf_var_dict[var_2_c],varAssignement,var_2_c)
                break  
        var_2_1 = inv_trans_dict[var_2,i]
        var_1_1 = inv_trans_dict[var_1,j]

        deltaE2 = -Eval_Delta_Energy(pbf, pbf_var_dict[var_1_0],varAssignement,var_1_0)
        varAssignement[var_1_0] = 0
        deltaE2 -= Eval_Delta_Energy(pbf, pbf_var_dict[var_2_0],varAssignement,var_2_0)
        varAssignement[var_2_0] = 0 
        deltaE2 -= Eval_Delta_Energy(pbf, pbf_var_dict[inv_trans_dict[var_2,i]],varAssignement,inv_trans_dict[var_2,i]) 
        varAssignement[var_2_1] = 1    
        deltaE2 -= Eval_Delta_Energy(pbf, pbf_var_dict[inv_trans_dict[var_1,j]],varAssignement,inv_trans_dict[var_1,j])
        varAssignement[var_1_1] = 1  
    

        if deltaE2 > 0:
            if allow_anyway(deltaE2, T) == 0:
                #switch back
                varAssignement[var_1_0] = 1 
                varAssignement[var_2_0] = 1 
                varAssignement[var_2_1] = 0 
                varAssignement[var_1_1] = 0  
                deltaE2 = 0
        Energy += deltaE2
        vals.append(Energy)
        if track_steps:
            if vals[step]< Minimum:
                Minimum = vals[step] 
            Ts.append(T)
        #test if eval correct
        if step == 0: vals2=[]
        vals2.append(evalPBF(pbf,varAssignement))               

    return Energy, varAssignement,vals,Minimum,Min_varAssignement,Ts,vars