from Funcs_Optimizers import *

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
    if 1:
        # Binary Cluster graph
        path = r"/home/lino/Dokumente/GitHub/fujitsuclone/TSP_data/ALL_tsp/a280.tsp" #linux
        #path = r"/Users/lino/Documents/python/fujitsuclone/TSP_data/ALL_tsp/EUR101.tsp" #mac
        print("Graph Clustering:")
        print("starting feature generation")
        start_time_gen =time.time()
        start_time_clustering=start_time_gen
        x_1 = 0
        x_2 = 10
        y_1 = 0
        y_2 = 5
        
        border_split = 60
        
        coords=read_in_csv_TSP(path)
        cluster = int(np.ceil(np.log2(len(coords)/border_split)))
        Groups = GraphPartitioning(coords,cluster,10,diff_border=5,steps=500)    
        
        fig2 = make_subplots(rows=1,cols=1)
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
        fig2.update_layout(title_text="Graph Partitioning with " + str(len(Groups)) + "Groups, generated in "+str(time.time() -start_time_clustering)+ " seconds")

        fig2.show()