from Funcs_Optimizers import *

if __name__ == "__main__":
    #parameters
    num_MC = 25
    steps=300
    variables = 300
    degree = 2

    Offset_increase = 100
    start_seed=42
    Anz = 10
    
    cooling_param = ["linear", 100,0.001]
    #cooling_param = ["auto", 5000,250,4.64254]
    #cooling_param = ["exponential", 1000000,1000000]
    for _ in range(15):
        cooling_param = ["logarithmic", int(variables/2),0]
        for _ in range(Anz):
            seed_gen = random.uniform(0,1000)
            seed_rand =[]
            random.seed(seed_gen)
            for i in range(num_MC):
                seed_rand.append(random.uniform(0,1000))
            
        #if 1:   #normal pbf solver

            print("starting feature generation")
            start_time_gen =time.time() 
            print("variables = " + str(variables))
            pbf = createPoly(variables,degree,1,seed=seed_gen)
            pbf=Sort_pbf(pbf,2)
            pbf_var_dict = createPolyDict(pbf,variables) 
            print("feature generation time:" + str(time.time() - start_time_gen))
                
            
            #pbf_min_solver(pbf,pbf_var_dict,"simulatedAnnealing",6000,50,cooling_param,seed_rand,seed_gen,visual_inst=True,save_csv=False)
            #pbf_min_solver(pbf,pbf_var_dict,"scaAnnealing",200,1,cooling_param,[seed_rand[0]],seed_gen,visual_inst=True,save_csv=True)
            #pbf_min_solver(pbf,pbf_var_dict,"digitalAnnealing_parallel",steps,num_MC,cooling_param,seed_rand,seed_gen,visual_inst=True,offset_increase_rate=Offset_increase,save_csv=False,save_addinfo=True)
            Min_VarAss,Min,TrajectoriesSA,result_List=pbf_min_solver(pbf,pbf_var_dict,"simulatedAnnealing",steps,num_MC,cooling_param,seed_rand,seed_gen,visual_inst=False,
                        offset_increase_rate=Offset_increase,save_addinfo=False, save_csv=True,random_start=False)  
            Min_VarAss,Min,TrajectoriesDA,result_List=pbf_min_solver(pbf,pbf_var_dict,"digitalAnnealing",steps,num_MC,cooling_param,seed_rand,seed_gen,visual_inst=False,
                        offset_increase_rate=Offset_increase,save_addinfo=False, save_csv=True,random_start=False)         
        variables += 100
            #Plot_Trajec_nice([TrajectoriesSA,TrajectoriesDA])
            #pbf_min_solver(pbf,pbf_var_dict,"digitalAnnealing_parallel",steps,num_MC,cooling_param,seed_rand,seed_gen, save_addinfo=False,
            #              visual_inst=True,offset_increase_rate=Offset_increase,save_csv=False,random_start=False)
            #variables += 100
