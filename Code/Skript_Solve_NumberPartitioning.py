from Funcs_Optimizers import *
from Funcs_pbfGenerators import GenerateNumberPartitioningpbf
if __name__ == "__main__":
    #parameters
    num_MC = 10
    steps=100

    variables = 500
    degree = 2
    fig_global=1
    Offset_increase = 100000000
    cooling_param = ["linear", 1000,100000]
    cooling_param = ["logarithmic", 10000000,0]
    #cooling_param = ["exponential", 1000000,1000000]
    seed_gen = 76243    
    seed_rand =[]
    random.seed(seed_gen)
    for i in range(num_MC):
        seed_rand.append(random.uniform(0,1000))
    if 1: #number partitioning
        numbers = Generate_Number_Partitioning_List(100,1000)
        pbf = GenerateNumberPartitioningpbf(numbers)
        pbf_var_dict = createPolyDict(pbf,len(numbers)) 
        problem = "NP"
        name_run = "800dwadawda0;1000"
        Min_VarAss,Mins,Trajectories,result_List = pbf_min_solver(pbf,pbf_var_dict,"digitalAnnealing",steps,num_MC,cooling_param,seed_rand,
                                        seed_gen,offset_increase_rate=Offset_increase,save_csv=False,save_addinfo=True,visual_inst=True)    
        Plot_Trajec_nice([Trajectories])
        #test procedure
        if 0:
            start_time_eval=time.time()
            #pbf_min_solver(pbf,pbf_var_dict,"digitalAnnealing_parallel",steps,num_MC,cooling_param,seed_rand,seed_gen,visual_inst=True,offset_increase_rate=Offset_increase,save_csv=False)
            Min_VarAss,Mins,Trajectories,result_List = pbf_min_solver(pbf,pbf_var_dict,"digitalAnnealing",steps,num_MC,cooling_param,seed_rand,seed_gen,offset_increase_rate=Offset_increase,save_csv=True,save_addinfo=True)         
            Save_Results_As_List(problem, "DA", name_run, len(list(pbf.keys())), time.time()-start_time_eval, steps, cooling_param, Mins)
            Min_VarAss,Mins,Trajectories,result_List = pbf_min_solver(pbf,pbf_var_dict,"simulatedAnnealing",steps,num_MC,cooling_param,seed_rand,seed_gen,offset_increase_rate=Offset_increase,save_csv=True,save_addinfo=True)      
            start_time_eval=time.time()
            Save_Results_As_List(problem, "SA", name_run, len(list(pbf.keys())), time.time()-start_time_eval, steps, cooling_param, Mins)  
    
    if 1:
        for j in range(num_MC):       
            GroupA=[]
            GroupB=[]
            for i in range(len(Min_VarAss[0])):
                if Min_VarAss[j][i]==0:
                    GroupA.append(numbers[i])
                else:
                    GroupB.append(numbers[i]) 
            print("Partitioning MC Trial "+str(j))
            print("sum Group A="+str(sum(GroupA)))
            print("sum Group B="+str(sum(GroupB)))
        #diffrence
            print("Diffrence:" +str(abs(sum(GroupA)-sum(GroupB))))