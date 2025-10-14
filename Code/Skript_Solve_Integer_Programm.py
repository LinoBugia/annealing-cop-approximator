from Funcs_Optimizers import *
from Funcs_pbfGenerators import IP_to_pbf
if __name__ == "__main__":
    #parameters
    num_MC = 200
    steps=50

    variables = 500
    degree = 2

    Offset_increase = 100000000

    cooling_param = ["linear", 1000,100000]
    cooling_param = ["logarithmic", 10000000,0]
    #cooling_param = ["exponential", 1000000,1000000]
    seed_gen = 76243
    seed_rand =[]
    random.seed(seed_gen)
    for i in range(num_MC):
        seed_rand.append(random.uniform(0,1000))
    if 1:   #IP or BIP solver
        print("starting feature generation")
        start_time_gen =time.time() 
        if 1:
            c = np.array([3,5,8,1])
            b= np.array([4,5,3])
            S = np.matrix(
                [[3,2,1,-1],
                [2,1,3,-1],
                [1,1,1,-1]]
            )
        elif 1:
            c = np.array([2,3,5,1,3,2])
            b= np.array([8])
            S = np.matrix(
                [[4,-1,-2,3,5,7]])       
        elif 1:
            c = np.array([3,5,2,1])
            b= np.array([4,5,3])
            S = np.matrix(
                [[3,2,1,-1],
                [2,-1,3,1],
                [1,-1,1,1]])
        anz_bit=3
        #pbf = BIP_to_pbf(S,b,c)            
        pbf = IP_to_pbf(S,b,c,anz_bit)
        pbf=Sort_pbf(pbf,2)
        pbf_var_dict = createPolyDict(pbf,c.shape[0]*anz_bit) 
        print("feature generation time:" + str(time.time() - start_time_gen))
        #pbf_min_solver(pbf,pbf_var_dict,"simulatedAnnealing",6000,50,cooling_param,seed_rand,seed_gen,visual_inst=True,save_csv=False)
        #pbf_min_solver(pbf,pbf_var_dict,"scaAnnealing",200,1,cooling_param,[seed_rand[0]],seed_gen,visual_inst=True,save_csv=True)
        #pbf_min_solver(pbf,pbf_var_dict,"digitalAnnealing_parallel",steps,num_MC,cooling_param,seed_rand,seed_gen,visual_inst=True,offset_increase_rate=Offset_increase,save_csv=False)
        Min_VarAss,Mins,Trajectories,result_List = pbf_min_solver(pbf,pbf_var_dict,"digitalAnnealing",steps,num_MC,cooling_param,seed_rand,seed_gen,visual_inst=True,save_addinfo=True,offset_increase_rate=Offset_increase,save_csv=False,random_start=True) 
        best = 0
        bests = []
        for i in range(num_MC):
            print("num_MC:" +str(i))
            print(Mins[i],evalPBF(pbf,Min_VarAss[i]))
            for k in range(c.shape[0]):
                #print(str(k*np.sqrt(len(input))))
                print(Min_VarAss[i][k*int(anz_bit):(k+1)*int(anz_bit)])
            costfunc = 0
            for j in range(c.shape[0]):
                for b in range(anz_bit): #(anz_bit):
                    costfunc +=c[j]* Min_VarAss[i][j*anz_bit+b]*pow(2,b)
            
            print("costfunc" + str(-costfunc))
            print("penalty" + str(Mins[i] + costfunc))
            if Mins[i] + costfunc == 0:
                costfunc > best
                best = costfunc
                bests.append(Min_VarAss[i])
        
        print("_____________________")
        if bests ==[]: print("no best without penalty violation")
        else:
            for i in range(len(bests)): 
                print("Best"+str(i))
                for k in range(c.shape[0]):
                        print(bests[i][k*int(anz_bit):(k+1)*int(anz_bit)])
            print(best)