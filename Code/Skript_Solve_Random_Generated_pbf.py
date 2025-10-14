from Funcs_Optimizers import *

if __name__ == "__main__":
    #parameters
    num_MC = 10
    steps=1000
    variables = 400
    degree = 2

    Offset_increase = 10000

    cooling_param = ["linear", 100,0.001]
    cooling_param = ["logarithmic", 20,0]
    #cooling_param = ["auto", 5000,250,4.64254]
    #cooling_param = ["exponential", 1000000,1000000]
    seed_gen = 76243
    seed_rand =[]
    random.seed(seed_gen)
    for i in range(num_MC):
        seed_rand.append(random.uniform(0,1000))
        
    if 1:   #normal pbf solver
    #for _ in range(10):
        print("starting feature generation")
        start_time_gen =time.time() 
        print("variables = " + str(variables))
        pbf = createPoly(variables,degree,1,seed=seed_gen)
        pbf=Sort_pbf(pbf,2)
        pbf_var_dict = createPolyDict(pbf,variables) 
        print("feature generation time:" + str(time.time() - start_time_gen))
        
        
        start_time_test = time.time()
        found = (variables-3,variables-2) in list(pbf.keys())
        print(time.time()-start_time_test)   
        
        start_time_test = time.time()
        found = (variables-3,variables-2) 
        print(time.time()-start_time_test)   
            
        
        #pbf_min_solver(pbf,pbf_var_dict,"simulatedAnnealing",6000,50,cooling_param,seed_rand,seed_gen,visual_inst=True,save_csv=False)
        #pbf_min_solver(pbf,pbf_var_dict,"scaAnnealing",200,1,cooling_param,[seed_rand[0]],seed_gen,visual_inst=True,save_csv=True)
        #pbf_min_solver(pbf,pbf_var_dict,"digitalAnnealing_parallel",steps,num_MC,cooling_param,seed_rand,seed_gen,visual_inst=True,offset_increase_rate=Offset_increase,save_csv=False,save_addinfo=True)
        Min_VarAss,Min,TrajectoriesSA,result_List=pbf_min_solver(pbf,pbf_var_dict,"simulatedAnnealing",steps,num_MC,cooling_param,seed_rand,seed_gen,visual_inst=True,
                       offset_increase_rate=Offset_increase,save_addinfo=True, save_csv=False,random_start=False)  
        Min_VarAss,Min,TrajectoriesDA,result_List=pbf_min_solver(pbf,pbf_var_dict,"digitalAnnealing",steps,num_MC,cooling_param,seed_rand,seed_gen,visual_inst=True,
                       offset_increase_rate=Offset_increase,save_addinfo=True, save_csv=False,random_start=False)         
        
        Plot_Trajec_nice([TrajectoriesSA,TrajectoriesDA])
        #pbf_min_solver(pbf,pbf_var_dict,"digitalAnnealing_parallel",steps,num_MC,cooling_param,seed_rand,seed_gen, save_addinfo=False,
        #              visual_inst=True,offset_increase_rate=Offset_increase,save_csv=False,random_start=False) 
        #variables += 100
        