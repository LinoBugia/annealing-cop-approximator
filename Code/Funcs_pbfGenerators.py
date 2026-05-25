from Funcs_Annealing2 import *

# Functions used to generate pbfs

# Input parameters defining the pbf
# Output pbf


def Generate_3SAT_pbf(path, inv=False):
    """
    Liest eine DIMACS 3-SAT Instanz und kodiert sie als PBF.
    Energie = Anzahl verletzter Klauseln (Minimum = 0 bei erfuellbarer Instanz).
    inv=True: negiert alle Koeffizienten.
    """
    f = -1 if inv else 1

    # --- DIMACS parsen + Metadaten sammeln ---
    clauses = []
    meta = {"horn": None, "forced": None, "mixed_sat": None, "clause_length": None}
    num_vars = None
    num_clauses_header = None

    with open(path) as file:
        for line in file:
            line_stripped = line.strip()

            # Kommentarzeilen: Metadaten extrahieren
            if line_stripped.startswith('c'):
                low = line_stripped.lower()
                if 'horn?' in low:
                    meta["horn"] = low.split('horn?')[1].strip()
                elif 'forced?' in low:
                    meta["forced"] = low.split('forced?')[1].strip()
                elif 'mixed sat?' in low:
                    meta["mixed_sat"] = low.split('mixed sat?')[1].strip()
                elif 'clause length' in low:
                    meta["clause_length"] = low.split('=')[1].strip()
                continue

            # Problem-Zeile
            if line_stripped.startswith('p'):
                parts = line_stripped.split()
                num_vars = int(parts[2])
                num_clauses_header = int(parts[3])
                continue

            if line_stripped in ('%', '0', ''):
                continue

            lits = list(map(int, line_stripped.split()))
            if lits[-1] == 0:
                lits = lits[:-1]
            if lits:
                clauses.append(lits)

    # --- Header ausgeben ---
    ratio = len(clauses) / num_vars if num_vars else float('nan')
    forced_str = meta["forced"] if meta["forced"] else "unknown"
    sat_status = "SAT guaranteed" if forced_str == "yes" else \
                 "UNSAT likely"   if ratio > 4.267 else \
                 "SAT likely"

    print("=" * 50)
    print(f"  3-SAT PBF Generator")
    print("=" * 50)
    print(f"  File         : {path}")
    print(f"  Variables    : {num_vars}")
    print(f"  Clauses      : {len(clauses)}  (header: {num_clauses_header})")
    print(f"  Ratio m/n    : {ratio:.4f}  (phase transition: 4.267)")
    print(f"  Forced SAT   : {forced_str}")
    print(f"  Horn         : {meta['horn'] or 'unknown'}")
    print(f"  Mixed SAT    : {meta['mixed_sat'] or 'unknown'}")
    print(f"  Inverted PBF : {inv}")
    print(f"  Status est.  : {sat_status}")
    print("=" * 50)

    # --- PBF aufbauen ---
    from itertools import combinations
    pbf = {}

    for lits in clauses:
        pos_idxs = [v - 1 for v in lits if v > 0]
        neg_idxs = [(-v) - 1 for v in lits if v < 0]

        for r in range(len(pos_idxs) + 1):
            for subset in combinations(pos_idxs, r):
                sign = (-1) ** r
                key = tuple(sorted(neg_idxs + list(subset)))
                pbf = Add_entry_pbf(pbf, key, f * sign)

    print(f"  PBF terms    : {len(pbf)}")
    print(f"  Constant ()  : {pbf.get((), 0)}")
    print("=" * 50 + "\n")

    return pbf, clauses, len(clauses),num_vars

    
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
def createPoly_negative(variables, degree, density=1.0, seed=42):
    """Wie createPoly aber alle Koeffizienten negativ — Ferromagnet"""
    random.seed(seed)
    out = dict()
    varList = list(range(variables))
    for i in range(1, degree + 1):
        for m in combinations(varList, i):
            if random.random() < density:
                out[tuple(m)] = -(random.random()*random.uniform(1,256) + random.random())
                # KEIN Vorzeichenwechsel mehr
    return out    

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