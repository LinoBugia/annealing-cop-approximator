from Funcs_Optimizers import *
import plotly.graph_objects as go

# Ising model approximator
from Funcs_pbfGenerators import Generate_2D_Ising_pbf

if __name__ == "__main__":
    #parameters
    num_MC = 1
    steps=200000 #multiple of 100 for visualizati‚on
    
    number_particles = 60
    J=-1
    h=0
    degree = 2

    cooling_param=("constant",1)
    cooling_param=("logarithmic",1)

    Offset_increase = 100

    # Generate Ising Schedule:
    dim = "2D"
    
    seed_gen = 76243
    seed_rand =[]
    random.seed(seed_gen)
    for i in range(num_MC):
        seed_rand.append(random.uniform(0,1000))
        
    print("starting feature generation")
    start_time_gen =time.time() 
    print("variables = " + str(number_particles))
    
    pbf,trans_dict,inv_trans_dict = Generate_2D_Ising_pbf("2D",number_particles,J,h)
    
    pbf=Sort_pbf(pbf,2)
    pbf_var_dict = createPolyDict(pbf,pow(number_particles,2)) 
       
    
    print(evalPBF_List(pbf,[0,1]*round(pow(number_particles,2)/2)) ,evalPBF_List(pbf,[1]*round(pow(number_particles,2))) )
    print("feature generation time:" + str(time.time() - start_time_gen))
    
    a,b,Trajectories_1sa,result_List_1sa=pbf_min_solver(pbf,pbf_var_dict,"simulatedAnnealing",steps,num_MC,cooling_param,seed_rand,seed_gen,visual_inst=True,offset_increase_rate=Offset_increase,save_addinfo=True, save_csv=False,random_start=True,Ising = True)   
    a,b,Trajectories_1da,result_List_1da=pbf_min_solver(pbf,pbf_var_dict,"digitalAnnealing",steps,num_MC,cooling_param,seed_rand,seed_gen,visual_inst=True,offset_increase_rate=Offset_increase,save_addinfo=True, save_csv=False,random_start=True,Ising = True)  

    pbf,trans_dict,inv_trans_dict = Generate_2D_Ising_pbf("2D",number_particles,-1,3)
    
    pbf=Sort_pbf(pbf,2)
    pbf_var_dict = createPolyDict(pbf,pow(number_particles,2)) 
       
    
    print(evalPBF_List(pbf,[0,1]*round(pow(number_particles,2)/2)) ,evalPBF_List(pbf,[1]*round(pow(number_particles,2))) )
    print("feature generation time:" + str(time.time() - start_time_gen))
    
    #a,b,Trajectories_2sa,result_List_2sa=pbf_min_solver(pbf,pbf_var_dict,"simulatedAnnealing",steps,10,cooling_param,seed_rand,seed_gen,visual_inst=True,offset_increase_rate=Offset_increase,save_addinfo=True, save_csv=False,random_start=True,Ising = True)   
    #a,b,Trajectories_2da,result_List_2da=pbf_min_solver(pbf,pbf_var_dict,"digitalAnnealing",steps,10,cooling_param,seed_rand,seed_gen,visual_inst=True,offset_increase_rate=Offset_increase,save_addinfo=True, save_csv=False,random_start=True,Ising = True)  
    #print(result_List)
    #Plot_Trajec_nice([Trajectories_1sa,Trajectories_1da])
    #Plot_Trajec_nice([Trajectories_2sa,Trajectories_2da])
    #print(result_List)
    Output =[] 
    if 0:
        for i in range(int(len(result_List)/2)):
            #if np.mod(i,20)==0:
            i=i+int(len(result_List)/2)-1
            if np.mod(i,100)==0:
                print("__________________Number__"+str(i)+"________Energy__"+str(Trajectories[0][i])+"_____________")
                for j in range(number_particles):
                    print(result_List[i][j*number_particles:(j+1)*number_particles])
    
   
    #validate  
    list=[] 
    
    for k in range(8):

        # Create a sample 2D binary image (e.g., 10x10 grid of 0s and 1s)
        
        image_data = np.random.randint(0, 2, size=(number_particles, number_particles))
        for i in range(pow(number_particles,2)):
            if np.mod(k,2)== 0:
                image_data[trans_dict[i]] = result_List_1sa[int(steps*k/8)][i]
            else:
                image_data[trans_dict[i]] = result_List_1da[int(steps*k/8)][i]
        # Plot using heatmap (suitable for binary images)
        fig = go.Figure(data=go.Heatmap(
            z=image_data,
            colorscale=[[0, 'white'], [1, 'black']],  # 0 = white, 1 = black
            showscale=False
        ))

        fig.update_layout(
            title=' ',
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        # Lock aspect ratios
        fig.update_layout(
            title="Square Binary Images",
            height=500,
            width=500,
            xaxis=dict(scaleanchor="y", showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            xaxis2=dict(scaleanchor="y2", showgrid=False, showticklabels=False),
            yaxis2=dict(showgrid=False, showticklabels=False)
        )
        fig.show()
        # Create a sample 2D binary image (e.g., 10x10 grid of 0s and 1s)
        
    
    if 1:
        for j in range(int(number_particles)):
            for i in range(int(number_particles/2)):
                if np.mod(j,2):
                    list.append(0)
                    list.append(1)
                else: 
                    list.append(1)
                    list.append(0)
    print("EnergiesPattern_010101:"+str(evalPBF_List(pbf,list)))
    print("EnergiesPattern_111111:"+str(evalPBF_List(pbf,[1]*round(pow(number_particles,2))) ))
    print("EnergiesPattern_000000:"+str(evalPBF_List(pbf,[0]*round(pow(number_particles,2))) ))
    
    