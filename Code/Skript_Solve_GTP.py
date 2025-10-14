from Funcs_Optimizers import *
from scipy.optimize import milp, LinearConstraint, Bounds
from Funcs_pbfGenerators import IP_to_pbf,BIP_to_pbf
if __name__ == "__main__":
    #parameters
    num_MC = 10
    steps=20000
    fig_global = 1
    seed_gen_GTP = 42
    Offset_increase = 100
    num_goods= 15
    fig_global = 1
    anz_bit = int(math.log(num_goods,2))
    
    r=3
    s=3
    print("starting feature generation")
    start_time_gen =time.time() 
    #anz_bit = 6
    A,b,c = Generate_GTP(num_goods,r,s,seed_gen_GTP)
    c=-c
    print("GTP generation time:" + str(time.time() - start_time_gen))
    start_time_gen =time.time() 
    pbf = IP_to_pbf(A,b,c,anz_bit,1)
    pbf_var_dict = createPolyDict(pbf,int(r*s*anz_bit)) 
    print("pbf generation time:" + str(time.time() - start_time_gen))
    
    cooling_param = ["auto", 5000,250,300]
    #cooling_param = ["logarithmic", 10000000000,0]
    #cooling_param = ["exponential", 1,1.5]
    seed_gen = 76243
    seed_rand =[]
    random.seed(seed_gen)
    for i in range(num_MC):
        seed_rand.append(random.uniform(0,1000))
        
    Min_VarAss,Mins,Trajectories,result_List = pbf_min_solver(pbf,pbf_var_dict,"digitalAnnealing",steps,num_MC,cooling_param,seed_rand,seed_gen,visual_inst=True,save_addinfo=True,offset_increase_rate=Offset_increase,save_csv=False,random_start=False) 
    best = 0
    bests = []
    for i in range(num_MC):
        print("num_MC:" +str(i))
        print(Mins[i],evalPBF(pbf,Min_VarAss[i]))
        if 0:
            for k in range(c.shape[0]):
                #print(str(k*np.sqrt(len(input))))
                print(Min_VarAss[i][k*int(anz_bit):(k+1)*int(anz_bit)])
        costfunc = 0
        for j in range(c.shape[0]):
            for l in range(anz_bit): #(anz_bit):
                costfunc +=c[j]* Min_VarAss[i][j*anz_bit+l]*pow(2,l)
        
        print("costfunc" + str(costfunc))
        print("penalty" + str(Mins[i] - costfunc))
        if abs(Mins[i] - costfunc) < pow(10,-4):
            costfunc > best
            best = - costfunc
            best_num = i
            bests.append([Min_VarAss[i],Mins[i],i])
    
    print("_____________________")
    if bests ==[]: print("no best without penalty violation")
    else:
        for i in range(len(bests)): 
            print("Best "+str(i) + " Number "+str(bests[i][2]))
            if 0:
                for k in range(c.shape[0]):
                        print(bests[i][0][k*int(anz_bit):(k+1)*int(anz_bit)])
            print(Mins[bests[i][2]])
        print("total best" +str(best_num)+": "+ str(best))

if bests!=[]:
    print("Annealer:" + str(Min_VarAss[best_num]))
    print("sol_ann"+str(evalPBF_List(pbf,Min_VarAss[best_num])))

# simplex solution:
constraints = LinearConstraint(A, b, b)
bounds = Bounds(0, np.inf)
integrality = np.ones_like(c)

res = milp(c=c, constraints=constraints, bounds=bounds , integrality=integrality)
print(res.x)
print("milp:"+str(res.fun))


sol=[]
for i in range(len(list(res.x))):
    x = format(int(res.x[i]), 'b')
    for j in range(anz_bit-len(x)):
        sol.append(0)
    for j in range(len(x)):
        sol.append(int(x[j]))
        
print("sol_milp: "+str(evalPBF_List(pbf,list(sol))))

#https://github.com/bstabler/TransportationNetworks
#https://logistik.bwl.uni-mainz.de/forschung/benchmarks/
