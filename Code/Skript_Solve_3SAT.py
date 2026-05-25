from Funcs_Optimizers import *
from itertools import groupby

from Funcs_pbfGenerators import Generate_3SAT_pbf
# 3 SAT SOLVER      
# 
# 
# 
def dpll(clauses, assignment=None):
    if assignment is None:
        assignment = {}
    
    # Unit propagation
    changed = True
    while changed:
        changed = False
        for c in clauses:
            satisfied = any(
                (l > 0 and assignment.get(abs(l)) == 1) or
                (l < 0 and assignment.get(abs(l)) == 0)
                for l in c
            )
            if satisfied:
                continue
            unset = [l for l in c if abs(l) not in assignment]
            if not unset:
                return None  # Konflikt
            if len(unset) == 1:
                l = unset[0]
                assignment = {**assignment, abs(l): (1 if l > 0 else 0)}
                changed = True

    # Alle Klauseln erfuellt?
    if all(
        any(
            (l > 0 and assignment.get(abs(l)) == 1) or
            (l < 0 and assignment.get(abs(l)) == 0)
            for l in c
        )
        for c in clauses
    ):
        return assignment

    # Branch auf erste ungesetzte Variable
    v = next(abs(l) for c in clauses for l in c if abs(l) not in assignment)
    for val in [1, 0]:
        result = dpll(clauses, {**assignment, v: val})
        if result is not None:
            return result
    return None

if __name__ == "__main__":
    #parameters
    num_MC_1 =200
    num_MC_2 = 200
    steps_1=10000
    steps_2=10000

    Offset_increase = 0

    printoutput = True

    #cooling_param = ["auto", 5000,250,4.64254]
    #cooling_param = ["exponential", 1000000,1000000]
    seed_gen = 76243
    seed_rand =[]
    random.seed(seed_gen)
    for i in range(num_MC_1):
        seed_rand.append(random.uniform(0,1000))
# -> data download: setshttps://www.cs.ubc.ca/~hoos/SATLIB/benchm.html#:~:text=Uniform%20Random%2D3%2DSAT%2C,100%20instances%2C%20all%20sat/unsat 
    #set path and 
    #set path and 
    path =  "/Users/lino/Documents/python/AnnealingCopApproximator/3SAT_DATA/uf50-218"
    #variables = 250
    #path= "/Users/lino/Documents/python/fujitsuclone/3SAT_DATA/UUF250.1065.100_satifyable/uuf250-015.cnf"
    #variables = 250
    #path =  "/Users/lino/Documents/python/fujitsuclone/3SAT_DATA/uf200-860"
    #variables = 200
    #path= "/Users/lino/Documents/python/fujitsuclone/3SAT_data/CBS_k3_n100_m403_b10_0.cnf"
    #variables = 100
    file_path = "/Users/lino/Documents/python/fujitsuclone/Code/data_thesis_SAT.txt"
    problems = ["uf50-0129.cnf"]
    #x_opt = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1]

    # Als dict: 0-basierte Indizes -> bool
    #x_opt_dict = {i: bool(v) for i, v in enumerate(x_opt)}
    for problem in problems:   
        if 0: # debug
            # Nach Generate_3SAT_pbf und DPLL:

            pbf, clauses, num_clauses,_ = Generate_3SAT_pbf(os.path.join(path,problem),inv=False)

            # DPLL auf clauses (bereits geparst, 1-basiert mit Vorzeichen)
            sol = dpll(clauses)

            if sol:
                n = max(abs(l) for c in clauses for l in c)
                x = [sol.get(i+1, 0) for i in range(n)]  # n aus p-Zeile
                print("SAT-Loesung: x =", x)
                
                energy = 0
                for key, coeff in pbf.items():
                    term = coeff
                    for idx in key:
                        term *= x[idx]
                    energy += term
                #evalPBF_List(pbf,sol)
                print(f"PBF Energie bei SAT-Loesung : {energy}")
                print(f"Erwartet                    : 0")
                print(f"Bug in PBF-Kodierung        : {energy != 0}")
            else:
                print("UNSAT")
        #cooling_param = ["logarithmic", 4.423,0]
        #cooling_param = ["logarithmic", 6.2310754,0]
        cooling_param = ["constant", 0.3,0]
        if 1:
            Min_VarAssisgnment=[]
        for i in range(num_MC_2):
            seed_rand.append(random.uniform(0,1000))
        pbf2,df,num_clauses, variables = Generate_3SAT_pbf(os.path.join(path,problem),inv=False)
        print(num_clauses)
        print("starting feature generation")
        start_time_gen =time.time() 
        #pbf = createPoly(variables,degree,1,seed=seed_gen)
        pbf2=Sort_pbf(pbf2,3)
        pbf_var_dict2 = createPolyDict(pbf2, variables) 
        print("feature generation time:" + str(time.time() - start_time_gen))

        Start=time.time() 
        if 1:
            Min_VarAss,Mins,Trajectories,result_List=pbf_min_solver(pbf2,pbf_var_dict2,"simulatedAnnealing",steps_2,num_MC_2,cooling_param,seed_rand,seed_gen,
                visual_inst=printoutput,offset_increase_rate=Offset_increase,save_addinfo=False, save_csv=False,
                initial_varAssignement_pre=Min_VarAssisgnment,random_start=True)           
            exec_time_SA = time.time()-Start
            acc_SA = 0
            for i in range(num_MC_2):
                if Mins[i]==0:
                    acc_SA = acc_SA +1  
        

    if 1:
        Start=time.time()
        Min_VarAss,Mins,Trajectories,result_List=pbf_min_solver(pbf2,pbf_var_dict2,"digitalAnnealing",steps_2,num_MC_2,cooling_param,seed_rand,seed_gen,
            visual_inst=printoutput,offset_increase_rate=Offset_increase,save_addinfo=False, save_csv=False,
            initial_varAssignement_pre=Min_VarAssisgnment,random_start=False)          
        exec_time_DA = time.time()-Start
        acc_DA=0
        for i in range(num_MC_2):
            if Mins[i]==0:
                acc_DA = acc_DA +1      
    
        with open(file_path, "a") as csvfile:
            csvwriter =csv.writer(csvfile)
            row = [problem, exec_time_SA,exec_time_DA, acc_SA,acc_DA,steps_2,int(steps_2),Offset_increase] #TODO
            csvwriter.writerow(row)    
        Offset_increase += 500
      
    
    if 0:# complete enumeration
        pbf,df,num_clauses = Generate_3SAT_pbf(path,inv=False)
        #pbf = createPoly(variables,degree,1,seed=seed_gen)
        pbf=Sort_pbf(pbf,3)
        pbf_var_dict = createPolyDict(pbf, variables) 
        Min_CE = 100
        for i in range(pow(2,variables)):
            n=i
            z = []
            while(n>0):
                a=n%2
                z.append(a)
                n=n//2
            for j in range(variables-len(z)):
                z.append(0)
            z.reverse()
            
            result =evalPBF_List(pbf,z)
            if result <= Min_CE:
                Min_CE = result
                Abs_Minimum = z
            print("Best" + str(Min_CE) + ", Num" +str(i) + ", Bitstring" + str(z))
        print("best: "+str(Min_CE) + ", bistring: " + str(Abs_Minimum))
    #pbf_min_solver(pbf,pbf_var_dict,"simulatedAnnealing",6000,50,cooling_param,seed_rand,seed_gen,visual_inst=True,save_csv=False)
    #pbf_min_solver(pbf,pbf_var_dict,"scaAnnealing",200,1,cooling_param,[seed_rand[0]],seed_gen,visual_inst=True,save_csv=True)

