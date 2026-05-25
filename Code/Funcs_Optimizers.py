from Funcs_Annealers import *
from Funcs_TSP import *
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time
import datetime
import os
import csv
#import plotly.express as px
import plotly.graph_objects as go
from Funcs_lkh import *
from Funcs_pbfGenerators import GenerateGraphBinaryClustering_pbf

import plotly.io as pio
pio.renderers.default = "browser"
def VisualizeRuns(pbf,type_alg, num_MC,offset_increase_rate,Trajectories,Mins,ExecTimes,T,Qs,Min_varAssignements,initial_varAssignment,trans_dict=dict(),coords=[]):
    if type_alg == "digitalAnnealing_TSP":
        rows =6
        cols =2
        subplot_titles=("Energy", "Delta Energy","Temp", "Exec Times [s]","Mins","E_Offset", "Solutions")
        specs=[
                [{"colspan": 2}, None],
                [{"colspan": 2}, None],
                [{"colspan": 2}, None],
                [{},{}],
                [{"colspan": 2}, None],
                [{"colspan": 2}, None]]
        row_heights=[0.6, 0.1,0.1,0.1,0.1,0.2]
    elif type_alg == "scaAnnealing":
        rows =5
        cols =2
        subplot_titles=("Energy", "Delta Energy","Temp", "Exec Times [s]","Mins","Qs")
        specs=[
                [{"colspan": 2}, None],
                [{"colspan": 2}, None],
                [{"colspan": 2}, None],
                [{},{}],
                [{"colspan": 2}, None]]
        row_heights=[0.6, 0.1,0.1,0.1,0.1]
    elif type_alg == "digitalAnnealing" or type_alg == "digitalAnnealing_old":
        rows =5
        cols =2
        subplot_titles=("Energy", "Delta Energy","Temp", "Exec Times [s]","Mins","E_Offset")
        specs=[
                [{"colspan": 2}, None],
                [{"colspan": 2}, None],
                [{"colspan": 2}, None],
                [{},{}],
                [{"colspan": 2}, None]]
        row_heights=[0.6, 0.1,0.1,0.1,0.1]          
    else:
        rows =4
        cols =2
        subplot_titles=("Energy", "Delta Energy","Temp", "Exec Times [s]","Mins")
        specs=[
                [{"colspan": 2}, None],
                [{"colspan": 2}, None],
                [{"colspan": 2}, None],
                [{},{}]]
        row_heights=[0.6, 0.1,0.1,0.1]  

    fig = make_subplots(rows=rows,cols=cols,
                        subplot_titles=subplot_titles,
                        specs=specs,
                        row_heights=row_heights
                        )
    
    fig.update_layout(title_text="Evaluation " + str(type_alg) +", Optimization variables = " + str(len(initial_varAssignment.keys())) +", size pbf: " +str(len(pbf.keys())))
    colors = []
    for i in range(100):
        k = np.mod(i,23)
        colors.append(plotly.colors.qualitative.Dark24_r[k])
    for i in range(num_MC):
        csv2 = pd.DataFrame(np.array(Trajectories[i]).transpose())
        fig.add_trace(go.Scatter(x=csv2.index, y=csv2[0],name="MC"+str(i),legendgroup="MC"+str(i),marker=dict(color=colors[np.mod(i,23)])),row=1,col=1)
        deltaE =csv2[0].diff()
        fig.add_trace(go.Scatter(x=csv2.index, y=deltaE,showlegend=False,legendgroup="MC"+str(i),marker=dict(color=colors[np.mod(i,23)])),row=2,col=1)

        if type_alg == "digitalAnnealing" or type_alg == "digitalAnnealing_TSP" or type_alg == "digitalAnnealing_parallel":
            E_Offsets=[]
            for j in range(len(deltaE)-1):
                if deltaE[j] == 0:
                    E_Offset += offset_increase_rate
                else:
                    E_Offset =0
                E_Offsets.append(E_Offset)
            fig.add_trace(go.Scatter(x=csv2.index, y=E_Offsets,marker=dict(color=colors[np.mod(i,23)]),name="MC"+str(i),showlegend=False,legendgroup="MC"+str(i)),row=5,col=1) 
    if type_alg == "scaAnnealing":
        fig.add_trace(go.Scatter(x=csv2.index, y=Qs,name = "Qs",marker=dict(color=colors[np.mod(i,23)]),showlegend=False),row=5,col=1) 
    if type_alg ==  "digitalAnnealing_TSP":
        for i in range(num_MC):
            order = Convert_var_Assignment_to_order(Min_varAssignements[i],trans_dict)
            ordered_coords=[]
            for j in order:
                ordered_coords.append(coords[j])
            x,y=np.hsplit(np.array(ordered_coords),2)
            x = x.flatten()
            y = y.flatten()
            fig.add_trace(
                go.Scatter(
                    mode="lines+markers" ,
                    showlegend=False,
                    legendgroup="MC"+str(i),
                    marker=dict(color=colors[np.mod(i,23)]),
                    x=x,
                    y=y),row=6,col=1
                    )
    fig.add_trace(go.Scatter(x=csv2.index, y=T,name="T",showlegend=False),row=3,col=1) 
    fig.add_trace(go.Scatter(x=list(range(num_MC)), y=Mins,name="Mins",showlegend=False),row=4,col=2)  
    fig.add_trace(go.Scatter(x=list(range(num_MC)), y=ExecTimes,name="Exec_Times",showlegend=False),row=4,col=1) 
    return fig

def pbf_min_solver(pbf : dict[tuple:float],pbf_var_dict: dict[int:tuple] ,
                   type_alg : str,steps :int,num_MC:int,cooling_param : list,
                   seed_rand :int, seed_gen_initial_varAssignment : int,
                   save_csv = False, save_addinfo = False, visual_inst = False,initial_varAssignement_pre=[],
                   inv_trans_dict=dict(),trans_dict = dict(),offset_increase_rate=0,coords=[],tour=[],random_start=False,Ising=False,visual_inst2=False):
    
    Trajectories = []
    ExecTimes = []
    Mins=[]
    Min_varAssignements = [] 
    Qs =[]
    result_List=[]
    time_init1=0
    time_init2=0
    #generate cooling schedule
    
    if type_alg=="scaAnnealing":
        Qs = Generate_Cooling_Schedule(["rising",0.01],steps)
    #generate var_assignement
    if type_alg == "digitalAnnealing_TSP":
        start_time_eval =time.time()
        N = int(np.sqrt(len(pbf_var_dict.keys())))
        initial_varAssignment = dict()
        for var in range(N*N):
            initial_varAssignment[var]= 0
        
        if tour == []:
            tour = list(range(N))
        #    random.shuffle(tour)
        k=0
        for var in tour:
            initial_varAssignment[(var)+(k*N)]=1
            k=k+1
        #Print_VarAssigmentCommandline(list(initial_varAssignment.values()))
        combs = list(combinations(range(N),2))
        #initialization of deltaE
        deltaE_init = [0]*len(combs)
        #test
        for var in range(len(combs)):
            var_1 = combs[var][0]
            var_2 = combs[var][1]
            for i in range(N):
                if initial_varAssignment[inv_trans_dict[var_1,i]] == 1:
                    var_1_0 = inv_trans_dict[var_1,i]
                    break
            for j in range(N):
                if initial_varAssignment[inv_trans_dict[var_2,j]] == 1:
                    var_2_0 = inv_trans_dict[var_2,j]
                    break  
            var_2_1 = inv_trans_dict[var_2,i]
            var_1_1 = inv_trans_dict[var_1,j]
            deltaE_init[var] = eval_deltaE_many_bitflips(pbf,pbf_var_dict, initial_varAssignment,[var_1_0,var_2_0,var_1_1,var_2_1])

        print("time evaluate delta E init: " + str(time.time() - start_time_eval))

    elif type_alg == "digitalAnnealing" or type_alg == "digitalAnnealing_parallel" or type_alg == "digitalAnnealing_old":
        if initial_varAssignement_pre!=[]:
            initial_varAssignment = dict(zip(range(len(pbf_var_dict.keys())),initial_varAssignement_pre))     
        else:
            initial_varAssignment =getInitialVarAssignement(pbf,seed_gen_initial_varAssignment)
        start_time_eval =time.time()
        #
        deltaE_init=[0]*len(initial_varAssignment)
        #deltaE_init = dict()
        for i in range(len(initial_varAssignment)):
            deltaE_init[i]=Eval_Delta_Energy(pbf, pbf_var_dict[i],initial_varAssignment,i)
        time_init1=time.time() - start_time_eval
        print("time evaluate delta E init: " + str(time_init1))
    elif type_alg =="simulatedAnnealing" or "scaAnnealing":
        
        if initial_varAssignement_pre!=[]:
            initial_varAssignment = dict(zip(range(len(pbf_var_dict.keys())),initial_varAssignement_pre))     
        else:
            initial_varAssignment =getInitialVarAssignement(pbf,seed_gen_initial_varAssignment)
        random.seed(seed_gen_initial_varAssignment)
    else:
        if initial_varAssignement_pre!=[]:
            initial_varAssignment = dict(zip(range(len(pbf_var_dict.keys())),initial_varAssignement_pre))     
        else:
            initial_varAssignment =getInitialVarAssignement(pbf,seed_gen_initial_varAssignment)
    print("Evaluation " + str(type_alg) +", Optimization variables = " + str(len(pbf_var_dict.keys())) +", size pbf: " +str(len(pbf.keys())))
    start_time_eval =time.time()
    E_init = evalPBF(pbf,initial_varAssignment)
    #print(E_init)
    time_init2=time.time() - start_time_eval
    print("time evaluate E init: " + str(time_init2))
    if 0:
        start_time_eval =time.time()
        Eval_Delta_Energy(pbf,pbf_var_dict[0],initial_varAssignment,1)
        time_Edelta = time.time() - start_time_eval
        print("time evaluate E delta: " + str(time_Edelta))
    #if type_alg=="digitalAnnealing_TSP":
    #    print("Approximated Time eval "+str(time_Edelta*steps*num_MC))
    #else:
    #    print("Approximated Time eval "+str(time_Edelta*steps*len(pbf_var_dict.keys())*num_MC))
    cores_to_use = 8
    if cooling_param[0]=="auto":
        cooling_param.append(E_init)
        cooling_param.append(pbf)
        cooling_param.append(pbf_var_dict)
        cooling_param.append(initial_varAssignment.copy())
        T = Generate_Cooling_Schedule(cooling_param,steps)
    else:    
        T = Generate_Cooling_Schedule(cooling_param,steps)
        
        
    if type_alg == "digitalAnnealing_parallel":
        with multiprocessing.Pool(processes=cores_to_use,initializer=pool_initializer,initargs=(pbf,)) as pool:
            for i in range(num_MC):
                start_time_round =time.time() 
                varAssignement = initial_varAssignment.copy()
                Output = DigitalAnnealing_parallel(pool,E_init,deltaE_init.copy(),pbf,pbf_var_dict,steps,varAssignement,T,save_addinfo,seed_rand[i],offset_increase_rate)
                endtime = time.time() - start_time_round
                Trajectories.append(Output[0])
                ExecTimes.append(endtime)
                Mins.append(np.min(Output[0]))
                if save_addinfo: Min_varAssignements.append(Output[1])
                print(str(i) +"-th MC trial: "+ str(endtime) + " sec, Average time rest execution " + str(np.mean(ExecTimes)*(num_MC-i-1)) + "sec, Min " +str(Mins[i]))
    else:
        #do the monte carlo trials:
        for i in range(num_MC):
            start_time_round =time.time() 
            if random_start or type_alg == "scaAnnealing" or type_alg == "simulatedAnnealing": 
                seed_gen_initial=random.randint(0,10000000000)
                initial_varAssignment =getInitialVarAssignement(pbf,seed_gen_initial)
                E_init = evalPBF(pbf,initial_varAssignment)    
                #deltaE_init=[0]*len(initial_varAssignment)
                #for z in range(len(deltaE_init)):
                #    deltaE_init[z]=Eval_Delta_Energy(pbf, pbf_var_dict[z],initial_varAssignment,z)
            varAssignement=initial_varAssignment.copy()
            #Print_VarAssigmentCommandline(list(varAssignement.values()))
            if type_alg == "simulatedAnnealing":
                Output = simulatedAnnealing(pbf,pbf_var_dict,steps,varAssignement,T,save_addinfo,seed_rand[i],Ising=Ising)
            elif type_alg == "digitalAnnealing":
                Output = DigitalAnnealing(E_init,deltaE_init.copy(),pbf,pbf_var_dict,steps,
                            varAssignement,T,save_addinfo,seed_rand[i],offset_increase_rate,Ising=Ising)
                #Output = DigitalAnnealing_old(E_init,pbf,pbf_var_dict,steps,varAssignement,T,1,seed_rand[i],offset_increase_rate)
            elif type_alg == "digitalAnnealing_old":
                Output = DigitalAnnealing_old(E_init,deltaE_init.copy(),pbf,pbf_var_dict,steps,
                            varAssignement,T,save_addinfo,seed_rand[i],offset_increase_rate)
            elif type_alg == "scaAnnealing":

                Output = SCA_Annealing(pbf,pbf_var_dict,steps,varAssignement,T,Qs,save_addinfo,seed_rand[i])
            elif type_alg ==  "digitalAnnealing_TSP":
                Output = digitalAnnealing_TSP(E_init,deltaE_init.copy(),pbf,pbf_var_dict,steps,varAssignement,T,seed_rand[i],inv_trans_dict,offset_increase_rate)


            endtime = time.time() - start_time_round
            ExecTimes.append(endtime)
            Trajectories.append(Output[0])
            #print(np.min(Output[0]))
            Mins.append(np.min(Output[0]))
            if save_addinfo: 
                Min_varAssignements.append(Output[1])
                #print("Min")
                #Print_VarAssigmentCommandline(Output[1])
            if 1:
                print(str(i) +"-th MC trial: "+ str(endtime) + " sec, Average time rest execution " + str(np.mean(ExecTimes)*(num_MC-i-1)) + "sec, Min " +str(Mins[i]))
    if save_csv:
        eval_directory = str(os.getcwd()) + "/Runs"
        if os.path.exists(eval_directory) == 0:
            os.mkdir(eval_directory)
        if os.path.exists(eval_directory+"/Evaluation_"+str(datetime.date.today())) == 0:
            os.mkdir(eval_directory+"/Evaluation_"+str(datetime.date.today())) 
        filename = 'Evaluation_'+str(datetime.date.today())+'.csv'

        if os.path.exists(os.path.join(eval_directory+"/Evaluation_"+str(datetime.date.today()),filename))==0:
            columns = ["type_alg","ID","time","timegen1","timegen2","bestMin","seed_gen","variables","degree","monomials_pbf","tpye_cooling","T_start","T_end"]
            df = pd.DataFrame(list(),columns=columns)
            df.to_csv(os.path.join(eval_directory+"/Evaluation_"+str(datetime.date.today()),filename))

        trajectories_directory = eval_directory+"/Evaluation_"+str(datetime.date.today()) +"/Trajectories"
        addinfo_directory =eval_directory+"/Evaluation_"+str(datetime.date.today()) + "/AddInfo"
        if os.path.exists(trajectories_directory) == 0:
            os.mkdir(trajectories_directory)
        if os.path.exists(addinfo_directory) == 0:
            os.mkdir(addinfo_directory)        
        ID_run = str(len(os.listdir(trajectories_directory))+1)
        #get ID
        #write Metaparameter line
        df = pd.DataFrame(list())
        for i in range(num_MC):
            df["MC"+str(i)]= Trajectories[i]        
        

        df.to_csv(trajectories_directory+"/Evaluation_"+str(datetime.date.today())+"_"+ID_run +"_trajectories.csv")  

        df = pd.DataFrame(list())
        for i in range(num_MC):
            df["MC"+str(i)]= Trajectories[i]        
        df = pd.DataFrame()
        df["Exec"]= ExecTimes
        df["rand_seed"] = seed_rand
        if save_addinfo:
            df["Min_var"] = Min_varAssignements
        df.to_csv(addinfo_directory+"/Evaluation_"+str(datetime.date.today())+"_"+ID_run +"_addinfo.csv")  
        with open(os.path.join(eval_directory+"/Evaluation_"+str(datetime.date.today()),filename), "a") as csvfile:
            csvwriter =csv.writer(csvfile)
            row = [type_alg,ID_run,np.sum(ExecTimes),time_init1,time_init2,np.min(Mins),seed_gen_initial_varAssignment,len(varAssignement.keys()),len(max(list(pbf.keys()),key =len)),len(pbf.keys()),cooling_param[0],cooling_param[1],cooling_param[2]] #TODO
            csvwriter.writerow(row)     
    if visual_inst:
        fig = VisualizeRuns(pbf,type_alg, num_MC,offset_increase_rate,Trajectories,Mins,ExecTimes,T,Qs,Min_varAssignements,initial_varAssignment,trans_dict=trans_dict,coords=coords)
        fig.show()
        if save_csv:
            fig.write_html(os.path.join(eval_directory+"/Evaluation_"+str(datetime.date.today()),'Evaluation_'+str(type_alg)+"_"+str(datetime.date.today())+"_"+ID_run +'.html'))
    if Ising: result_List =Output[2]
        
    
    return Min_varAssignements,Mins,Trajectories,result_List


def GraphPartitioning(coords,parts,num_MC_GP=1,diff_border=1,steps = 200):
    #Bisection_ordering=[]

    Groups2=[]
    Groups = [coords]
    k=0
    Offset_increase=1000
    visualize_inst=False
    seed_gen = random.uniform(1,1000)
    for t in range(parts):
        k=0
        Groups2=[]
        for Group in Groups:
            c=2*len(coords)/(t+1)
            #check if finished
            seed_rand_GP=[]
            for i in range(num_MC_GP):
                seed_rand_GP.append(random.uniform(0,1000))
            k=k+1
            start_time_gen = time.time()
            print("cluster" +str(t+1) + "/"+ str(parts)+ ", Number "+str(k)+ "/"+ str(len(Groups)))
            pbf= GenerateGraphBinaryClustering_pbf(Group)
            cooling_param = ["logarithmic", c,0]
            pbf_var_dict = createPolyDict(pbf,len(Group))
            print("size pbf: " +str(len(pbf.keys())))
            print("feature generation time:" + str(time.time() - start_time_gen))
            #if t== 0:visualize_inst=True
            Min_VarAss,Min,Trajectories,result_List= pbf_min_solver(pbf,pbf_var_dict,"digitalAnnealing",steps,num_MC_GP,cooling_param,seed_rand_GP,seed_gen,
                                            visual_inst=visualize_inst,offset_increase_rate=Offset_increase,save_csv=False,save_addinfo=True,random_start=True)      
            #visualize_inst=False
            Min_VarAss =Min_VarAss[int(np.argmin(Min))]
            GroupA=[]
            GroupB=[]
            for i in range(len(Min_VarAss)):
                if Min_VarAss[i]==0:
                    GroupA.append(Group[i])
                else:
                    GroupB.append(Group[i]) 
            diff =len(GroupA)-len(GroupB)
            if 1:
                if diff == 0:
                    1
                if diff < -diff_border:
                    diff = -diff
                    if np.mod(diff,2)==1:
                        diff = diff-1
                    for _ in range(int(diff/2)):
                        GroupA,GroupB = GroupA_TransferClosestPointGroupB(GroupA,GroupB)

                elif diff > diff_border:
                    diff = diff
                    if np.mod(diff,2)==1:
                        diff = diff-1
                    for _ in range(int(diff/2)):
                        GroupB,GroupA = GroupA_TransferClosestPointGroupB(GroupB,GroupA)                            
            Groups2.append(GroupA)
            Groups2.append(GroupB)
        #ordering = []
        for i in range(len(Groups2)):
            print(len(Groups2[i]))
        #    ordering.append(i)
        #Bisection_ordering.append(ordering)
        Groups = Groups2
    return Groups




if __name__ == "__main__":
    #parameters
    if 0:

########## ALTER CODE BACKUP ##########
        if 0: 
            cooling_param = ["logarithmic",10,0]  
            fig_end = make_subplots(rows=1,cols=1)
            fig_end.update_layout(title_text="Graph Partitioning with TSP")
            fig_end.update_layout(
                autosize=False,
                width=800,
                height=800,
                xaxis=dict(scaleanchor="y", scaleratio=1),
                yaxis=dict(scaleanchor="x", scaleratio=1),
            )
            ib=0
            list_ordered_coords=[]
            Mins=[]
            for Group in Groups:
                ib=ib+1
                print("Traveling Salesman Group " +str(ib)+"/" +str(len(Groups))+", number of cities =" +str(len(Group)))
                cooling_param = ["logarithmic",50,0]
                start_time_gen = time.time()
                Matrix = calculate_dist_Matrix(Group)
                Matrix = Matrix + Matrix.transpose()
                seed_rand_TSP =[]
                num_MC_TSP=1
                for i in range(num_MC_TSP):
                    seed_rand_TSP.append(random.uniform(0,1000))
                pbf = dict()
                trans_dict=dict()
                inv_trans_dict=dict()
                pbf, trans_dict, inv_trans_dict = convert_Matrix_to_pbf(Matrix,0)
                inital_tour = shortest_path_heuristic(Matrix)
                pbf_var_dict = createPolyDict(pbf,len(trans_dict.keys()))
                print("feature generation time:" + str(time.time() - start_time_gen))
                if 1:
                    
                    Min_VarAss,Min,Trajectories,result_List=pbf_min_solver(pbf,pbf_var_dict,"digitalAnnealing_TSP",1000,num_MC_TSP,cooling_param,seed_rand_TSP,seed_gen,
                            visual_inst=False,save_addinfo = True,inv_trans_dict=inv_trans_dict,trans_dict=trans_dict,coords=Group,offset_increase_rate=Offset_increase,save_csv=False,tour=inital_tour)  
                    print(cooling_param[1])
                    Min_VarAssisgnment= Min_VarAss[int(np.argmin(Min))]
                    Mins.append(min(Min))
                    order = Convert_var_Assignment_to_order(Min_VarAssisgnment,trans_dict)
                    #order = Tour_Clean_Up_Routeswitch(order,Matrix)
                    ordered_coords=[]
                    for j in order:
                        ordered_coords.append(Group[j])
                    x,y=np.hsplit(np.array(ordered_coords),2)
                    x = x.flatten()
                    y = y.flatten()
                    fig_end.add_trace(
                        go.Scatter(
                            mode="lines+markers" ,
                            showlegend=True,
                            x=x,
                            y=y),row=1,col=1
                            )
                    list_ordered_coords.append(ordered_coords)
            k=0
            required_merges=int(np.log2(len(list_ordered_coords)))
            fig_end.show()
            while len(list_ordered_coords)!=1:
                fig_intermediate = make_subplots(rows=1,cols=1)
                fig_intermediate.update_layout(title_text="Final route")
                fig_intermediate.update_layout(
                    autosize=False,
                    width=800,
                    height=800,
                    xaxis=dict(scaleanchor="y", scaleratio=1),
                    yaxis=dict(scaleanchor="x", scaleratio=1),
                )
                k=k+1
                list_ordered_coords2=[]
                # zero mean the coord
                mean_x=0
                mean_y=0
                coords_mean= []
                for i in range(len(list_ordered_coords)):
                    for j in range(len(list_ordered_coords[i])):    
                        mean_x+=list_ordered_coords[i][j][0]
                        mean_y+=list_ordered_coords[i][j][1]
                    mean_x= mean_x/len(list_ordered_coords[i])
                    mean_y= mean_y/len(list_ordered_coords[i])
                    coords_mean.append([mean_x,mean_y])
                OutputGP=GraphPartitioning(coords_mean,cluster-k,num_MC_GP=50,steps=150) 
                c=0
                Mins2=[]
                for pair in OutputGP:  
                    c=c+1
                    for t in range(len(coords_mean)):
                        if pair[0]==coords_mean[t]:
                            number1= t
                        elif pair[1]==coords_mean[t]:
                            number2=t
                    connected = Connect_2_Tours_Closest_point(list_ordered_coords[number1],list_ordered_coords[number2])
                    #final TSP optimization
                    
                    
                    print("Traveling Salesman Merge" +str(k)+"/" +str(required_merges)+"Group "+str(c)+"/"+str(len(OutputGP))+", number of cities =" +str(len(connected)))
                    cooling_param = ["logarithmic_step",1*(Mins[number1]+Mins[number2]+15),1]
                    start_time_gen = time.time()
                    Matrix = calculate_dist_Matrix(connected)
                    Matrix = Matrix + Matrix.transpose()
                    seed_rand_TSP_Merge =[]
                    num_MC_TSP_Merge=1
                    for i in range(num_MC_TSP_Merge):
                        seed_rand_TSP_Merge.append(random.uniform(0,1000))
                    pbf = dict()
                    trans_dict=dict()
                    inv_trans_dict=dict()
                    pbf, trans_dict, inv_trans_dict = convert_Matrix_to_pbf(Matrix,0)
                    initial_tour=[]
                    for j in range(len(connected)):
                        inital_tour.append(j)
                    pbf_var_dict = createPolyDict(pbf,len(trans_dict.keys()))
                    print("feature generation time:" + str(time.time() - start_time_gen))
                    if 1:
                        if k>2:
                            visualize= True
                        else:
                            visualize=False
                        Min_VarAss,Min,Trajectories,result_List=pbf_min_solver(pbf,pbf_var_dict,"digitalAnnealing_TSP",500,num_MC_TSP_Merge,cooling_param,seed_rand_TSP_Merge,seed_gen,
                                visual_inst=visualize,save_addinfo = True,inv_trans_dict=inv_trans_dict,trans_dict=trans_dict,coords=Group,offset_increase_rate=Offset_increase*k,save_csv=False,tour=inital_tour)  
                        Min_VarAssisgnment= Min_VarAss[int(np.argmin(Min))]
                        Mins2.append(min(Min))
                        order = Convert_var_Assignment_to_order(Min_VarAssisgnment,trans_dict)
                        #order = Tour_Clean_Up_Routeswitch(order,Matrix)
                        print(cooling_param[1])
                    else:
                        order = list(range(len(connected)))
                    connected_tour=[]
                    for j in order:
                        connected_tour.append(connected[j])
                    list_ordered_coords2.append(connected_tour)
                    x,y=np.hsplit(np.array(connected_tour),2)                          
                    x = x.flatten()
                    y = y.flatten()
                    fig_intermediate.add_trace(
                        go.Scatter(
                            mode="lines+markers" ,
                            showlegend=True,
                            x=x,
                            y=y),row=1,col=1
                            )
                
                fig_intermediate.show()
                Mins = Mins2
                list_ordered_coords=list_ordered_coords2
            if 0:
                fig_final = make_subplots(rows=1,cols=1)
                fig_final.update_layout(title_text="Final route")
                fig_final.update_layout(
                    autosize=False,
                    width=800,
                    height=800,
                    xaxis=dict(scaleanchor="y", scaleratio=1),
                    yaxis=dict(scaleanchor="x", scaleratio=1),
                )
                x,y=np.hsplit(np.array(list_ordered_coords[0]),2)                          
                x = x.flatten()
                y = y.flatten()
                fig_final.add_trace(
                    go.Scatter(
                        mode="lines+markers" ,
                        showlegend=True,
                        x=x,
                        y=y),row=1,col=1
                        )
                fig_final.show()
        print("Final execution time:" + str(time.time() - start_time_clustering))