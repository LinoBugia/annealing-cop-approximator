from Funcs_Optimizers import *
from itertools import groupby

from Funcs_pbfGenerators import Generate_3SAT_pbf
# 3 SAT SOLVER      
# 
# 
# 
if __name__ == "__main__":
    #parameters
    num_MC_1 =1000
    num_MC_2 = 100
    steps_1=100
    steps_2=10000

    Offset_increase = 10000


    cooling_param = ["linear", 100,0.001]
    cooling_param = ["logarithmic", 2,0]
    #cooling_param = ["auto", 5000,250,4.64254]
    #cooling_param = ["exponential", 1000000,1000000]
    seed_gen = 76243
    seed_rand =[]
    random.seed(seed_gen)
    for i in range(num_MC_1):
        seed_rand.append(random.uniform(0,1000))
# -> data download: setshttps://www.cs.ubc.ca/~hoos/SATLIB/benchm.html#:~:text=Uniform%20Random%2D3%2DSAT%2C,100%20instances%2C%20all%20sat/unsat 
    #set path and 
    path =  "/Users/lino/Documents/python/fujitsuclone/uf20-91/uf20-03.cnf"
    variables = 20
    #path= "/Users/lino/Documents/python/fujitsuclone/3SAT_DATA/UUF250.1065.100_satifyable/uuf250-015.cnf"
    #variables = 250
    #path =  "/Users/lino/Documents/python/fujitsuclone/uf20-91/Test.cnf"
    #variables = 10
    #path= "/Users/lino/Documents/python/fujitsuclone/3SAT_data/CBS_k3_n100_m403_b10_0.cnf"
    #variables = 100
       
    if 1:
        Min_VarAssisgnment=[]  
    
    for i in range(num_MC_2):
        seed_rand.append(random.uniform(0,1000))
    pbf2,df,num_clauses = Generate_3SAT_pbf(path,inv=False)
    print(num_clauses)
    print("starting feature generation")
    start_time_gen =time.time() 
    #pbf = createPoly(variables,degree,1,seed=seed_gen)
    pbf2=Sort_pbf(pbf2,3)
    pbf_var_dict2 = createPolyDict(pbf2, variables) 
    print("feature generation time:" + str(time.time() - start_time_gen))
    
    pbf_min_solver(pbf2,pbf_var_dict2,"digitalAnnealing",steps_2,num_MC_2,cooling_param,seed_rand,seed_gen,
        visual_inst=True,offset_increase_rate=Offset_increase,save_addinfo=True, save_csv=False,
        initial_varAssignement_pre=Min_VarAssisgnment,random_start=False)          
    
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

