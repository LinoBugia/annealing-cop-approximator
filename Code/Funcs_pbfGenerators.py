from Funcs_Annealing2 import *

# Functions used to generate pbfs

# Input parameters defining the pbf
# Output pbf

def Generate_3SAT_pbf(path,inv=False):
    
    df = pd.read_csv(path,skiprows=8,names=["a","b","c","0"], delim_whitespace=True)
    
    if inv == True:
        f = -1
    else:
        f = 1
    df = df[:-2].astype(int)
    
    
    pbf = dict()
    num_clauses = 0
    
    for index,row in df.iterrows():
        # 4 cases 
        a,b,c,d=row
        list = [a,b,c,]
        list.sort()
        
        for i in range(len(list)):
            if list[i] > 0:
                list[i]=list[i]-1
            else:
                list[i]=list[i]+1
                
                
                
        if a>= 0 and b>= 0 and c>= 0:
        # + + +
            pbf = Add_entry_pbf(pbf,(),+1*f)
            pbf = Add_entry_pbf(pbf,(list[0],),-1*f)
            pbf = Add_entry_pbf(pbf,(list[1],),-1*f)
            pbf = Add_entry_pbf(pbf,(list[2],),-1*f)
            pbf = Add_entry_pbf(pbf,tuple(sorted((list[0],list[1]))),+1*f)
            pbf = Add_entry_pbf(pbf,tuple(sorted((list[0],list[2]))),+1*f)
            pbf = Add_entry_pbf(pbf,tuple(sorted((list[1],list[2]))),+1*f)
            pbf = Add_entry_pbf(pbf,tuple(sorted((list[0],list[1],list[2]))),-1*f)
        # - - -
        elif a< 0 and b< 0 and c< 0:
            pbf = Add_entry_pbf(pbf,tuple(sorted((-list[0],-list[1],-list[2]))),+1*f)
        # - + +       
        elif list[1]>0:
            list[0]=-list[0]
            pbf = Add_entry_pbf(pbf,(list[0],),+1*f)     
            pbf = Add_entry_pbf(pbf,tuple(sorted((list[0],list[1]))),-1*f)     
            pbf = Add_entry_pbf(pbf,tuple(sorted((list[0],list[2]))),-1*f)  
            pbf = Add_entry_pbf(pbf,tuple(sorted((list[0],list[1],list[2]))),+1*f)     
        # - - +     
        else: 
            list[0]=-list[0]
            list[1]=-list[1]
            pbf = Add_entry_pbf(pbf,tuple(sorted((list[0],list[1]))),+1*f)     
            pbf = Add_entry_pbf(pbf,tuple(sorted((list[0],list[1],list[2]))),-1*f)     

        num_clauses+=1 
    return pbf,df,num_clauses

    
def Generate_2D_Ising_pbf(dim:str,N:int,J:float,h:float):
    
    
    trans_dict = dict()
    t= 0
    for i in range(N):
        for j in range(N):
            trans_dict[t]=(i,j)
            t+=1
    inv_trans_dict =dict(zip(trans_dict.values(), trans_dict.keys()))
    
    pbf = dict()
    if h!=0:
        for i in range(int(N*N)):
            pbf[(i,)] = -2*h
            pbf = Add_entry_pbf(pbf,(),-1)
    
    
    for i in range(N):
        for j in range(N):
            # i++ j
            pbf = Add_entry_pbf(pbf,(inv_trans_dict[(int(np.mod(i+1,N)),j)],inv_trans_dict[(i,j)]),-J/2)
            pbf = Add_entry_pbf(pbf,(inv_trans_dict[(int(np.mod(i+1,N)),j)],),+J/4)
            pbf = Add_entry_pbf(pbf,(inv_trans_dict[(i,j)],),+J/4)
            pbf = Add_entry_pbf(pbf,(),-J/2)
            # i-- j
            pbf = Add_entry_pbf(pbf,(inv_trans_dict[(int(np.mod(i-1,N)),j)],inv_trans_dict[(i,j)]),-J/2)
            pbf = Add_entry_pbf(pbf,(inv_trans_dict[(int(np.mod(i-1,N)),j)],),+J/4)
            pbf = Add_entry_pbf(pbf,(inv_trans_dict[(i,j)],),+J/16)
            pbf = Add_entry_pbf(pbf,(),-J/2)
            # i j++
            pbf = Add_entry_pbf(pbf,(inv_trans_dict[(i,int(np.mod(j+1,N)))],inv_trans_dict[(i,j)]),-J/2)
            pbf = Add_entry_pbf(pbf,(inv_trans_dict[(i,int(np.mod(j+1,N)))],),+J/4)
            pbf = Add_entry_pbf(pbf,(inv_trans_dict[(i,j)],),+J/4)
            pbf = Add_entry_pbf(pbf,(),-J/2)           
            # i j--
            pbf = Add_entry_pbf(pbf,(inv_trans_dict[(i,int(np.mod(j-1,N)))],inv_trans_dict[(i,j)]),-J/2)
            pbf = Add_entry_pbf(pbf,(inv_trans_dict[(i,int(np.mod(j-1,N)))],),+J/4)
            pbf = Add_entry_pbf(pbf,(inv_trans_dict[(i,j)],),+J/4)
            pbf = Add_entry_pbf(pbf,(),-J/2)           
    return pbf, trans_dict, inv_trans_dict


def BP_to_pbf(S:np.array,b:np.array,c:np.array):
    """
    This function converts an Integer Programm in Standardform 
    min c^T z , Az=b, z>= 0
    
    to a pbf via the terms
    
    H_c = c^T z
    H_A = B(Az-b)^2
    z = (x_1,x_2,...x_anz_bit) (>0)
    
    Input: 
        A, b, c, anz_bit
    
    Output:
        pbf
        
    """
    
    
    # get dimensions
    
    dim_c = c.shape[0]
    dim_b = b.shape[0]
    
    if S.shape[1]!=dim_c or S.shape[0]!=dim_b:
        raise ValueError("Dimensions wrong")
    
    pbf = dict()
    
    const = 0
    for i in range(b.shape[0]):
        const += pow(b[i],2)
    
    pbf[()] = const
    if 1:
        #Kostenfunktionsvekor pbf Verrechnung
        for i in range(dim_c):
            pbf[(i,)] = - c[i]

    #Matrix Gleichung Penalty Term  
    # sum_ij (a_ij x_i- b_i )^2 = sum_ij (a_ij - 2 a_ij b_i) *2^k sum_k x_{(j*bits)+k}  )
    #if 1:
    B=c.shape[0]
    for i in range(dim_b):
        for j in range(dim_c):
                if pbf.__contains__((j,)):
                    pbf[(j,)] += (pow(S[(i,j)],2) - 2* S[(i,j)]*b[i])*B
                else:
                    pbf[(j,)] = (pow(S[(i,j)],2)- 2* S[(i,j)]*b[i])*B

    for i in range(dim_b):
            for j in range(dim_c):
                for k in range(dim_c):
                    if k != j:
                        if pbf.__contains__((j,k)):
                            pbf[(j,k)] += S[(i,j)]*S[(i,k)]*B
                        else:
                            pbf[(j,k)] = S[(i,j)]*S[(i,k)]*B


    return pbf


def BIP_to_pbf(S:np.array,b:np.array,c:np.array):
    """
    This function converts an Integer Programm in Standardform 
    min c^T z , Az=b, z>= 0
    
    to a pbf via the terms
    
    H_c = c^T z
    H_A = B(Az-b)^2
    z = (x_1,x_2,...x_anz_bit) (>0)
    
    Input: 
        A, b, c, anz_bit
    
    Output:
        pbf
        
    """
    A= 1
    B= float(c.shape[0]*np.max(c))
    # get dimensions
    
    dim_c = c.shape[0]
    dim_b = b.shape[0]
    
    if S.shape[1]!=dim_c or S.shape[0]!=dim_b:
        raise ValueError("Dimensions wrong")
    

    
    pbf = dict()
    
    const = 0
    for i in range(b.shape[0]):
        const += pow(b[i],2)
    
    pbf[()] = const*B
    if 1:
        #Kostenfunktionsvekor pbf Verrechnung
        for i in range(dim_c):
            pbf[(i,)] = - c[i]*A

    #Matrix Gleichung Penalty Term  
    # sum_ij (a_ij x_i- b_i )^2 = sum_ij (a_ij - 2 a_ij b_i) *2^k sum_k x_{(j*bits)+k}  )
    #if 1:
    for i in range(dim_b):
        for j in range(dim_c):
                if pbf.__contains__((j,)):
                    pbf[(j,)] += (pow(S[(i,j)],2) - 2* S[(i,j)]*b[i])*B
                else:
                    pbf[(j,)] = (pow(S[(i,j)],2)- 2* S[(i,j)]*b[i])*B

    for i in range(dim_b):
            for j in range(dim_c):
                for k in range(dim_c):
                    if k != j:
                        if pbf.__contains__((j,k)):
                            pbf[(j,k)] += S[(i,j)]*S[(i,k)]*B
                        else:
                            pbf[(j,k)] = S[(i,j)]*S[(i,k)]*B

    return pbf

def IP_to_pbf(S:np.array,b:np.array,c:np.array,anz_bit:int,constant):
    """
    This function converts an Integer Programm in Standardform 
    min c^T z , Az=b, z>= 0
    
    to a pbf via the terms
    
    H_c = c^T z
    H_A = B(Az-b)^2
    z = (x_1,x_2,...x_anz_bit) (>0)
    
    Input: 
        A, b, c, anz_bit
    
    Output:
        pbf
        
    """
    
    
    # get dimensions
    
    dim_c = c.shape[0]
    dim_b = b.shape[0]
    
    if S.shape[0]!=dim_b or S.shape[1]!=dim_c:
        raise ValueError("Dimensions wrong")
    
    A=1
    total = 0
    for i in range(dim_c):
        max = np.min(c[(i,)],0)
        total -= max
         
        
    B= total/2
    pbf = dict()
    const = 0
    
    #A=1
    

    for i in range(b.shape[0]):
        const += pow(b[i],2)
    pbf[()] = const*B
    if 1:
        #Kostenfunktionsvekor pbf Verrechnung
        for i in range(dim_c):
            for j in range(anz_bit):
                pbf[(i*anz_bit+j,)] = c[i] *pow(2,j) *A


    #Matrix Gleichung Penalty Term  
    # sum_ij (a_ij x_i- b_i )^2 = sum_ij (a_ij - 2 a_ij b_i) *2^k sum_k x_{(j*bits)+k}  )
    if 1:
            
        for i in range(dim_b):
            for j in range(dim_c):
                for k in range(anz_bit):
                    if S[(i,j)] != 0:
                        if pbf.__contains__((j*anz_bit+k,)):
                            pbf[(j*anz_bit+k,)] += - 2* S[(i,j)]*b[i] *pow(2,k)*B
                        else:
                            pbf[(j*anz_bit+k,)] = - 2* S[(i,j)]*b[i] *pow(2,k)*B
    if 1:
        for i in range(dim_b):
                for j in range(dim_c):
                    for k in range(dim_c):
                        if S[(i,j)]!= 0 and S[(i,k)] != 0:
                                for b_1 in range(anz_bit):
                                    for b_2 in range(anz_bit):
                                            if pbf.__contains__((j*anz_bit+b_1,k*anz_bit+b_2)):
                                                pbf[(j*anz_bit+b_1,k*anz_bit+b_2)] += S[(i,j)]*S[(i,k)] *pow(2,b_1+b_2)*B
                                            else:
                                                pbf[(j*anz_bit+b_1,k*anz_bit+b_2)] = S[(i,j)]*S[(i,k)] *pow(2,b_1+b_2)*B
        #clean up
        
    iterate_list = list(pbf.keys())
    for key in iterate_list:
        if len(key)==2:
            if key[0]==key[1]:
                pbf[(key[0],)]+=pbf[key]
                del pbf[key]
    iterate_list = list(pbf.keys())         
    for key in iterate_list:
        if pbf[key] == 0:         
            del pbf[key]
    if 0:                            
    #tests 
        if 0:
            sol=[0]*anz_bit*dim_c
            sol[10+3]=1
            
        else:
            sol=[0]*anz_bit*dim_c       
            sol[0]=1
            sol[21]=1
        print(evalPBF_List(pbf,sol))
            
        for k in range(c.shape[0]):
            #print(str(k*np.sqrt(len(input))))
            
            print(sol[k*int(anz_bit):(k+1)*int(anz_bit)])
    return pbf




def convert_IBMQ_QUBO_to_pbf(IBMQ_QUBO):
    "this function converts a IBMQ QUBO to a sorted pbf in dictionary format "
    # extract linear and quadratic terms and some conversion for the right format
    pbf = dict()
    d1 = IBMQ_QUBO[0].objective.linear.to_dict()
    d3 = dict()
    d4 = dict()
    for i in d1.keys():
        key = (i,)
        d3[key]=d1[i]
    d2 =IBMQ_QUBO[0].objective.quadratic.to_dict()
    num_vars = len(IBMQ_QUBO[0].variables)
    for i in d2.keys():
        if i[0]==i[1]:
            if (i[0],) in d3.keys():
                d3[(i[0],)] = float(d3[(i[0],)]+d2[i])
            else:
                d3[(i[0],)] = d2[i]
        else:
            d4[i]=d2[i]
    #sort dictionary
    d3=dict(sorted(d3.items(), key=lambda x: x))
    d4 = dict(sorted(d4.items(), key=lambda x: x))
    #merge the dicts
    pbf = d3 | d4
    return pbf, num_vars

#Gen Functions
    

def GenerateNumberPartitioningpbf(numbers: list[float])-> dict[tuple:float]:

    if len(numbers)/2 == 0:
        raise ValueError("The numbers should be even")
    

    pbf = dict()

    #TODO
    pbf[()]=((sum(numbers))**2 )/2
    # expression (sum n_i* x_i)^2 = sum n_i + comb(n_i) 
    for i in range(len(numbers)):
        pbf[(i,)]=-4*numbers[i]*(sum(numbers)-numbers[i])
        
    
    combs = combinations(range(len(numbers)),2)
    
    for comb in combs:
        pbf[comb]=numbers[comb[0]]*numbers[comb[1]]*8
        #pbf[(comb[1],comb[0])]=numbers[comb[0]]*numbers[comb[1]]*4
    return pbf

def GenerateGraphBinaryClustering_pbf(coords):
    """
    This function creates a binary clustering pbf.
    """
    coords_mean=[]
    # zero mean the coord
    mean_x=0
    mean_y=0
    for i in range(len(coords)):
        mean_x+=coords[i][0]
        mean_y+=coords[i][1]
    mean_x= mean_x/len(coords)
    mean_y= mean_y/len(coords)
    mag = 0
    for i in range(len(coords)):
        
        coords_mean.append([coords[i][0]-mean_x,coords[i][1]-mean_y])
        vector=np.array([coords[i][0]-mean_x,coords[i][1]-mean_y])
        mag+=np.sqrt(vector.dot(vector))  
    x=np.asarray(coords_mean,dtype=float)
    #x = x.mean(axis=1)
    x=x*len(coords)/(mag)
    x=list(x)
    # Calculate Gram Matrix 
    Q=np.zeros((len(coords),len(coords)))
    for i in range(len(coords)):   
        for j in range(len(coords)):  
            Q[(i,j)]= np.dot(x[i],x[j])
            
    Q_row=np.sum(Q,axis=0)
    Q_col=np.sum(Q,axis=1)
    #create the pbf        
    pbf = dict()
    for i in range(len(coords)):
        pbf[(i,)] = Q_row[i]*Q_col[i]

    
    for i in range(len(coords)):
        for j in range(len(coords)):
            pbf[(i,j)]=-2*Q[(i,j)]
    
    return pbf