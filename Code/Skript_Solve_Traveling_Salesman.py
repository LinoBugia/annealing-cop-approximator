from Funcs_Optimizers import *

if __name__ == "__main__":
    #parameters
    num_MC = 200
    steps=50

    variables = 500
    degree = 2

    Offset_increase = 1000

    cooling_param = ["linear", 1000,100000]
    cooling_param = ["logarithmic", 10000000,0]
    #cooling_param = ["exponential", 1000000,1000000]
    seed_gen = 76243
    seed_rand =[]
    random.seed(seed_gen)
    for i in range(num_MC):
        seed_rand.append(random.uniform(0,1000))
    #TSP part 

    if 0:
        #TSP as pbf with penalty term
        path = "/home/lino/Dokumente/GitHub/fujitsuclone/TSP_data/ALL_tsp/EUR101.tsp"
        path = "/home/lino/Dokumente/GitHub/fujitsuclone/TSP_data/ALL_tsp/berlin52.tsp"
        path = r"/Users/lino/Documents/python/fujitsuclone/TSP_data/ALL_tsp/berlin52.tsp" #mac
        num_MC =500
        seed_rand =[]
        for i in range(num_MC):
            seed_rand.append(random.uniform(0,1000))
        print("starting feature generation")
        start_time_gen =time.time()
        x_1 = 0
        x_2 = 10
        y_1 = 0
        y_2 = 2
        steps = 300
        cooling_param = ["linear", 1000,2000] 
        cooling_param = ["logarithmic_step", 20000,1]
        coords= [[0,1],[0,2],[0,3],[3,3],[3,2],[3,0],[2,0],[1,0],[0,0]]
        #coords= [[0,0],[0,2],[2,2],[2,0],[3,0]]
        #coords=read_in_csv_TSP(path)
    
        coords = make_random_coords(x_1,x_2,y_1,y_2,10,seed = seed_gen)
        Matrix = calculate_dist_Matrix(coords)
        N=len(coords)
    

        pbf, trans_dict, inv_trans_dict = convert_Matrix_to_pbf(Matrix,penalty_term=1,a=1.1,b=1)
        pbf = Sort_pbf(pbf,2)
        #inital_tour = shortest_path_heuristic(Matrix,3)
        #initial_tour=list(range(N))
        pbf_var_dict = createPolyDict(pbf,len(trans_dict.keys()))
        print("size pbf: " +str(len(pbf.keys())))
        print("feature generation time:" + str(time.time() - start_time_gen))

        Min_VarAss,Min,Trajectories,result_List=pbf_min_solver(pbf,pbf_var_dict,"digitalAnnealing",steps,num_MC,cooling_param,seed_rand,seed_gen,random_start=True,
                    visual_inst=True,save_addinfo = True,inv_trans_dict=inv_trans_dict,trans_dict=trans_dict,coords=coords,offset_increase_rate=Offset_increase,save_csv=True)
        if 1:            
            fig_intermediate = make_subplots(rows=1,cols=2)
            fig_intermediate.update_layout(
                autosize=False,
                width=1400,
                height=700,
                xaxis=dict(scaleanchor="y", scaleratio=1),
                yaxis=dict(scaleanchor="x", scaleratio=1),
            )            
            
            k=0
            for Min_VarAssisgnment in Min_VarAss:
                order = Convert_var_Assignment_to_order(Min_VarAssisgnment,trans_dict)
                        #order = Tour_Clean_Up_Routeswitch(order,Matrix)
                connected_tour =[]
                for i in order:
                    connected_tour.append(coords[i])
                connected_tour.append(coords[0])
                x,y=np.hsplit(np.array(connected_tour),2)                          
                x = x.flatten()
                y = y.flatten()
                fig_intermediate.add_trace(
                    go.Scatter(
                        mode="lines+markers" ,
                        showlegend=True,
                        legendgroup=k,
                        name=str(k)+":"+str(order),
                        marker=dict(color=plotly.colors.qualitative.Dark24_r[np.mod(i,23)]),
                        x=x,
                        y=y),row=1,col=1
                        )  
                fig_intermediate.add_trace(
                    go.Scatter(
                        mode="lines+markers" ,
                        legendgroup=k,
                        showlegend=False,
                        name=str(k)+":"+str(order),
                        marker=dict(color=plotly.colors.qualitative.Dark24_r[np.mod(i,23)]),
                        y=[Min[k]],
                        x=[k]),row=1,col=2
                        )  
                k = k+1                 
            fig_intermediate.show()
    #TSP            DA Version
    if 0:
        path = "/home/lino/Dokumente/GitHub/fujitsuclone/TSP_data/ALL_tsp/d657.tsp"
        path = "/home/lino/Dokumente/GitHub/fujitsuclone/TSP_data/ALL_tsp/berlin52.tsp"
        path = r"/Users/lino/Documents/python/fujitsuclone/TSP_data/ALL_tsp/EUR101.tsp" #mac
        num_MC =10
        save_add_info =0
        seed_rand =[]
        for i in range(num_MC):
            seed_rand.append(random.uniform(0,1000))
        print("starting feature generation")
        start_time_gen =time.time()
        x_1 = 0
        x_2 = 10
        y_1 = 0
        y_2 = 2
        steps = 2000
        #cooling_param = ["linear", 1000,2000] 
        cooling_param = ["logarithmic", 20 ,1]
        N = 25
        #coords=read_in_csv_TSP(path)
        #coords= [[0,1],[0,2],[0,3],[3,3],[3,2],[3,0],[2,0],[1,0],[0,0]]
        #coords= [[0,0],[0,2],[2,2],[2,0]]
        coords = make_random_coords(x_1,x_2,y_1,y_2,N,seed = seed_gen)
        Matrix = calculate_dist_Matrix(coords)


    
        pbf, trans_dict, inv_trans_dict = convert_Matrix_to_pbf(Matrix,0)
        pbf = Sort_pbf(pbf,2)
        inital_tour = shortest_path_heuristic(Matrix)
        pbf_var_dict = createPolyDict(pbf,len(trans_dict.keys()))
        print("size pbf: " +str(len(pbf.keys())))
        print("feature generation time:" + str(time.time() - start_time_gen))
        if 1:
            pbf_min_solver(pbf,pbf_var_dict,"digitalAnnealing_TSP",steps,num_MC,cooling_param,seed_rand,seed_gen,
                    visual_inst=True,save_addinfo = True,inv_trans_dict=inv_trans_dict,trans_dict=trans_dict,coords=coords,offset_increase_rate=Offset_increase,save_csv=True,tour=inital_tour)
 
    #TSP with DA & lk combined
    if 0:
        path = "/home/lino/Dokumente/GitHub/fujitsuclone/TSP_data/ALL_tsp/d657.tsp"
        path = "/home/lino/Dokumente/GitHub/fujitsuclone/TSP_data/ALL_tsp/berlin52.tsp"
        path = r"/Users/lino/Documents/python/fujitsuclone/TSP_data/ALL_tsp/berlin52.tsp" #mac
        num_MC =5
        seed_rand =[]
        for i in range(num_MC):
            seed_rand.append(random.uniform(0,1000))
        print("starting feature generation")
        start_time_gen =time.time()
        x_1 = 0
        x_2 = 10
        y_1 = 0
        y_2 = 2
        
        steps_total = 10
        
        steps_tsp = 100
        steps_lkh = 500
        #cooling_param = ["linear", 1000,2000] 
        cooling_param = ["logarithmic", 8 ,1]
        N = 30
        #coords=read_in_csv_TSP(path)
        #coords= [[0,1],[0,2],[0,3],[3,3],[3,2],[3,0],[2,0],[1,0],[0,0]]
        #coords= [[0,0],[0,2],[2,2],[2,0]]
        
        
        #coords = make_random_coords(x_1,x_2,y_1,y_2,N,seed = seed_gen)
        Matrix = calculate_dist_Matrix(coords)


    
        pbf, trans_dict, inv_trans_dict = convert_Matrix_to_pbf(Matrix,0)
        pbf = Sort_pbf(pbf,2)
        inital_tour = shortest_path_heuristic(Matrix)
        pbf_var_dict = createPolyDict(pbf,len(trans_dict.keys()))
        print("size pbf: " +str(len(pbf.keys())))
        print("feature generation time:" + str(time.time() - start_time_gen))
        if 1:
            for i in range(steps_total):
                Min_VarAss,Min,Trajectories,result_List=pbf_min_solver(pbf,pbf_var_dict,"digitalAnnealing_TSP",steps_tsp,num_MC,cooling_param,seed_rand,seed_gen,
                    visual_inst=False,save_addinfo = True,inv_trans_dict=inv_trans_dict,trans_dict=trans_dict,coords=coords,offset_increase_rate=Offset_increase,save_csv=True,tour=inital_tour)         
                
                Min_VarAssisgnment= Min_VarAss[int(np.argmin(Min))]
                order = Convert_var_Assignment_to_order(Min_VarAssisgnment,trans_dict)
                #order = Tour_Clean_Up_Routeswitch(order,Matrix)
                ordered_coords=[]
                
                for j in order:
                    ordered_coords.append(coords[j])
                    
                ordered_coords,normal_best=solve_lkh(ordered_coords,runs=steps_lkh,shuffle=False)
                initial_tour=[]
                for i in range(len(coords)):
                    for coord in ordered_coords:
                        if coords[i]==coord:
                            initial_tour.append(i)
                            continue
            x,y=np.hsplit(np.array(ordered_coords),2)
            x = x.flatten()
            y = y.flatten()
            fig_visual = make_subplots(rows=2,cols=3)
            fig_end = make_subplots(rows=1,cols=1)
            
            fig_end.update_layout(
                autosize=False,
                width=800,
                height=800,
                xaxis=dict(scaleanchor="y", scaleratio=1),
                yaxis=dict(scaleanchor="x", scaleratio=1),
            )   
            
            fig_end.add_trace(
                go.Scatter(
                    mode="lines+markers" ,
                    showlegend=True,
                    x=x,
                    y=y),row=1,col=1
                    )  
            fig_visual.add_trace(
            go.Scatter(
                mode="lines+markers" ,
                showlegend=True,
                x=x,
                y=y),row=1,col=1
                )  
            fig_end.show()                   
                
    if 1:
        visual = False
        # Binary Cluster graph & lk TSP Solver
        #path = r"/home/lino/Dokumente/GitHub/fujitsuclone/TSP_data/ALL_tsp/a280.tsp" #linux
        #path = r"/Users/lino/Documents/python/fujitsuclone/TSP_data/ALL_tsp/EUR101.tsp" #mac
        #path = "/home/lino/Dokumente/GitHub/fujitsuclone/TSP_data/ALL_tsp/berlin52.tsp"
        #path = "home/lino/Dokumente/GitHub/fujitsuclone/TSP_data/ALL_tsp/d493.tsp"
        #
        #path = r"/home/lino/Dokumente/GitHub/fujitsuclone/TSP_data/ALL_tsp/rl.tsp"
        #path = r"/Users/lino/Documents/python/fujitsuclone/TSP_data/ALL_tsp/nrw1379.tsp" #mac
        #path = r"/Users/lino/Documents/python/fujitsuclone/TSP_data/ALL_tsp/u1060.tsp" #mac
        #path = r"/Users/lino/Documents/python/fujitsuclone/TSP_data/ALL_tsp/p654.tsp" #mac
        #path = r"/Users/lino/Documents/python/fujitsuclone/TSP_data/ALL_tsp/u724.tsp" #mac
        #path = r"/Users/lino/Documents/python/fujitsuclone/TSP_data/ALL_tsp/u2319.tsp" #mac
        #path = r"/Users/lino/Documents/python/fujitsuclone/TSP_data/ALL_tsp/d493.tsp" #mac
        path = r"/Users/lino/Documents/python/fujitsuclone/TSP_data/ALL_tsp/d1655.tsp" #mac
        #path = r"/Users/lino/Documents/python/fujitsuclone/TSP_data/ALL_tsp/u1060.tsp" #mac
        #path = r"/Users/lino/Documents/python/fujitsuclone/TSP_data/ALL_tsp/rl1889.tsp" #mac
        #path = r"/Users/lino/Documents/python/fujitsuclone/TSP_data/ALL_tsp/ali535.tsp"
        #path = r"/Users/lino/Documents/python/fujitsuclone/TSP_data/ALL_tsp/d1655.tsp"
        #path = r"/Users/lino/Documents/python/fujitsuclone/TSP_data/ALL_tsp/u2152.tsp"
        #path = r"/Users/lino/Documents/python/fujitsuclone/TSP_data/ALL_tsp/fl1577.tsp"
        print("TSP via Graph Clustering:")
        print("starting feature generation")
        start_time_gen =time.time()
        start_time_clustering=start_time_gen
        x_1 = 0
        x_2 = 10
        y_1 = 0
        y_2 = 5
        #coords = make_random_coords(0,100,0,100,N,seed = seed_gen)  
        border_split = 30
        
        coords=read_in_csv_TSP(path)
        cluster = int(np.ceil(np.log2(len(coords)/border_split)))
        Groups = GraphPartitioning(coords,cluster,3,diff_border=5,steps=1000)    
        #fig_visual = make_subplots(rows=2,cols=3,    vertical_spacing=0,  # no vertical gap between rows
        #    horizontal_spacing=0)  # no horizontal gap between columns)
        if visual:
            fig2 = make_subplots(rows=1,cols=1)
            fig2.update_layout(title_text="Graph Partitioning with" + str(len(Groups)))
            fig2.update_layout(
                autosize=False,
                width=800,
                height=800,
                xaxis=dict(scaleanchor="y", scaleratio=1),
                yaxis=dict(scaleanchor="x", scaleratio=1),
            )
            for Group in Groups:
                x,y=np.hsplit(np.array(Group),2)
                x = x.flatten()
                y = y.flatten()
                fig2.add_trace(go.Scatter(x=x,y=y,showlegend=True),row=1,col=1) 
                #fig_visual.add_trace(go.Scatter(x=x,y=y,showlegend=True),row=1,col=1) 
            fig2.show()
        list_ordered_coords=[]
        for Group in Groups:
            tour,best_com = solve_lkh(Group)
            list_ordered_coords.append(tour)
            #plot_tour(tour)
        if 1:
            k=0
            required_merges=int(np.log2(len(list_ordered_coords)))
            while len(list_ordered_coords)!=1:
                list_ordered_coords2=[]
                if visual:
                    fig_intermediate = make_subplots(rows=1,cols=1)
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
                OutputGP=GraphPartitioning(coords_mean,cluster-k,20,diff_border=0,steps=200) 
                for pair in OutputGP: 
                    if pair != []: 
                        for t in range(len(coords_mean)):
                            if pair[0]==coords_mean[t]:
                                number1= t
                            elif pair[1]==coords_mean[t]:
                                number2=t
                        connected = Connect_2_Tours_Closest_point(list_ordered_coords[number1],list_ordered_coords[number2])
                        improved_lkh,best_comb=solve_lkh(connected,runs=round(border_split/2)+10*k)
                        list_ordered_coords2.append(improved_lkh)

                        x,y=np.hsplit(np.array(improved_lkh),2)                          
                        x = x.flatten()
                        y = y.flatten()
                        if visual:
                            fig_intermediate.add_trace(
                                go.Scatter(
                                    mode="lines+markers" ,
                                    showlegend=True,
                                    x=x,
                                    y=y),row=1,col=1
                                    )
                        print(k)
                        col = int((k+1)%4)
                        row =int(np.floor((k-1)/2)+1)
                        if row ==2:
                            col = col+1
                        #fig_visual.add_trace(
                        #    go.Scatter(
                        #        mode="lines+markers" ,
                        #        showlegend=True,
                        #        x=x,
                        #        y=y),row=row,col = col
                        #        )
                if visual:
                    fig_intermediate.update_layout(title_text="Final route" +str(best_comb))
                    fig_intermediate.show()
                
                
                
                list_ordered_coords=list_ordered_coords2
        end_time_clustering = time.time()-start_time_clustering
        print(end_time_clustering)

        if 1:               #conventiional lk heuristic
            start_time_lkh = time.time()
            if visual:
                fig_end = make_subplots(rows=1,cols=1)
                
                fig_end.update_layout(
                    autosize=False,
                    width=800,
                    height=800,
                    xaxis=dict(scaleanchor="y", scaleratio=1),
                    yaxis=dict(scaleanchor="x", scaleratio=1),
                )
            normal_lkh = coords
            var_time = end_time_clustering - 1
            while var_time<end_time_clustering:

                normal_lkh,normal_best=solve_lkh(normal_lkh,runs=5,shuffle=False)
                
                var_time = time.time()-start_time_lkh


            coords = normal_lkh
            if visual:
                x,y=np.hsplit(np.array(coords),2)                          
                x = x.flatten()
                y = y.flatten()
                fig_end.add_trace(
                    go.Scatter(
                        mode="lines+markers" ,
                        showlegend=True,
                        x=x,
                        y=y),row=1,col=1
                        )  
                #fig_visual.add_trace(
                #    go.Scatter(
                #        mode="lines+markers" ,
                #        showlegend=True,
                #        x=x,
                #        y=y),row=2,col=3
                #        )              
                fig_end.update_layout(title_text="Final route " +str(normal_best)+" time: "+str(var_time))   
                fig_end.show()
            #coords = make_random_coords(x_1,x_2,y_1,y_2,300,seed = seed_gen)

    # safe output

    file_path = "TSP_GP_Output.txt"

    with open(file_path, "a") as csvfile:
        csvwriter =csv.writer(csvfile)
        row = [path,var_time,best_comb,normal_best,round(best_comb/normal_best,3)] #TODO
        csvwriter.writerow(row)   

if 0:
    fig_visual.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )

    fig_visual.update_layout(
        xaxis=dict(title=dict(font=dict(size=20)),tickfont=dict(size=16)),
        yaxis=dict(title=dict(font=dict(size=20)),showticklabels=False),
        legend=dict(font=dict(size=20))
    )
    fig_visual.update_layout(
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )    
    fig_visual.update_layout(
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            ticks='',
            title=None
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            ticks='',
            title=None
        ),
        showlegend=False,       # Hides the legend
        plot_bgcolor='white',   # Removes gray background
        paper_bgcolor='white',  # Removes outer background
        margin=dict(l=0, r=0, t=0, b=0)  # Optional: remove all margins
    )
    for axis in fig_visual.layout:
        if axis.startswith('xaxis') or axis.startswith('yaxis'):
            fig_visual.layout[axis].update(showticklabels=False)
    
    fig_visual.show()
    