# function related to the tensor evaluation of the QUBO

import tensorflow as tf
from Funcs_Annealing2 import *

def ConvertPBFtoTensorFormFull(pbf,max,degree = 4):
    """
     this function takes in a pbf as dictionary and returns it it converted as tensors. 
     Tensors are a generalization of Matrices and Vektors see: https://www.tensorflow.org/guide/tensor
    
    DIMENSIONS:
     degree = corresponds with Number of necessary Tensors = degree of pbf
     max = Number of variables per tensor = number of variables in pbf
    
     {(1,2,4): 3.4, (3,): 7, ...} (lets say degree = 3, max = 5)
     
     <-> 

     Tensorlist[0] = Tensor of shape (max) = vektor = 1-Tensor
     Tensorlist[1] = Tensor of shape (max,max) = Matrix = 2-Tensor
     Tensorlist[2] = Tensor of shape (max,max,max) = 3-Tensor
     ...
     Tensorlist[4] = Tensor of shape (max,max,max,max,max) = 5er-Tensor

    ASSIGNEMENT:

    (1,2,4): 3.4 
     <->
    Tensorlist[2].Index (1,2,4) = 3.4          
    
    Why should we do this?
    Advantagte: 
    no need to iterate over all monoms when evaluating the pbf. 
    You can target arbitrarly Columns/Rows/Matrixes/Parttensors etc. 
    This allows faster evaluation of specific parts of the pbf for detecting the change of the Hammiltonian when switching only one variable !
    """
    Tensor_List = []
    # Initialize Tensor variables 
    for i in range(1,degree+1):
        T = tf.Variable(tf.zeros([max]*i))
        Tensor_List.append(T)

    # iterate over pbf and assign the indices to the corresponding tensors
    for monomial in pbf:
        combinations = list(permutations(list(monomial)))
        for comb in combinations:
            Tensor_List[len(list(comb))-1][list(comb)].assign(pbf[monomial])
    # convert to tensor datatype. Cant be changed but operates faster.
    for i in range(degree):
        Tensor_List[i] = tf.convert_to_tensor(Tensor_List[i],dtype="float32")
    return Tensor_List

def ConvertPBFtoTensorForm(pbf,max,degree = 4):
    """
     this functtion takes in a pbf as dictionary and returns it it converted as tensors. 
     Tensors are a generalization of Matrices and Vektors see: https://www.tensorflow.org/guide/tensor
    
    DIMENSIONS:
     degree = corresponds with Number of necessary Tensors = degree of pbf
     max = Number of variables per tensor = number of variables in pbf
    
     {(1,2,4): 3.4, (3,): 7, ...} (lets say degree = 3, max = 5)
     
     <-> 

     Tensorlist[0] = Tensor of shape (max) = vektor = 1-Tensor
     Tensorlist[1] = Tensor of shape (max,max) = Matrix = 2-Tensor
     Tensorlist[2] = Tensor of shape (max,max,max) = 3-Tensor
     ...
     Tensorlist[4] = Tensor of shape (max,max,max,max,max) = 5er-Tensor

    ASSIGNEMENT:

    (1,2,4): 3.4 
     <->
    Tensorlist[2].Index (1,2,4) = 3.4          
    
    Why should we do this?
    Advantagte: 
    no need to iterate over all monoms when evaluating the pbf. 
    You can target arbitrarly Columns/Rows/Matrixes/Parttensors etc. 
    This allows faster evaluation of specific parts of the pbf for detecting the change of the Hammiltonian when switching only one variable !
    """
    Tensor_List = []
    # Initialize Tensor variables 
    for i in range(1,degree+1):
        T = tf.Variable(tf.zeros([max]*i))
        Tensor_List.append(T)

    # iterate over pbf and assign the indices to the corresponding tensors
    for monomial in pbf:
        #combinations = list(permutations(list(monomial)))
        #for comb in combinations:
        comb = monomial
        Tensor_List[len(list(comb))-1][list(comb)].assign(pbf[monomial])
    # convert to tensor datatype. Cant be changed but operates faster.
    for i in range(degree):
        Tensor_List[i] = tf.convert_to_tensor(Tensor_List[i],dtype="float32")
    return Tensor_List

def EvalPbfAsTensor(Tensor_List,varAssignment_vek):
    E=0
    #erstmal nur 2. grades
    for i in range(len(Tensor_List)):
        H = Tensor_List[i]
        for j in range(i+1):
            #H = H*varAssignment_vek
            #H=tf.np.tensordot(H, tf.cast(varAssignment_vek,float), name=None)
            H = tf.tensordot(H, tf.cast(varAssignment_vek,float),axes=1)
        E += float(H) 
    return E
def ConvertDicToVektor(varAssignement):
    varAssignment_vek = []
    for var in range(len(varAssignement)):
        varAssignment_vek.append(varAssignement[var])
    
    return varAssignment_vek

def EvalDeltaEAsTensor(Tensor_List,varAssignment,pos):
    #flip num in vektor
    varAssignment_c = varAssignment.copy()
    if varAssignment_c[pos]==0:
        varAssignment_c[pos]=1


    E=float(Tensor_List[0][pos])


    for i in range(1,len(Tensor_List)):
        T1 = Tensor_List[i]
        index=[]
        for j in range(1,i):
            index.append(slice(len(varAssignment_c)))
        index.append(pos)
        T2=T1[index]
        #print(T2)
        for j in range(1,i+1):
            T2 = tf.tensordot(T2, tf.convert_to_tensor(varAssignment_c,dtype=float),axes=1)
            #print(T2)

        E += float(T2) 
    if varAssignment[pos]==1:
        return E
    else:
        return -E 

def Eval_delta_E_as_QUBO(h,J,varAssignement):
    #input: varAssignement, Tensorlist
    deltaE = h+tf.tensordot(J, tf.convert_to_tensor(varAssignement,dtype=float),axes=1)
    #deltaE = tf.math.reduce_sum(deltaE,axis=0)
    multp_fac = -2*np.array(varAssignement) +1
    deltaE = tf.multiply(deltaE,tf.convert_to_tensor(multp_fac,dtype=float))
    #output: deltaE's
    return -deltaE

def Eval_delta_E_as_PUBO(Tensor_List,varAssignement):
    #input: varAssignement, Tensorlist
    multp_fac = -2*np.array(varAssignement) +1
    deltaE = Tensor_List[0]
    for i in range(1,len(Tensor_List)):
        J = Tensor_List[i]
        for j in range(i):
            J = tf.tensordot(J, tf.convert_to_tensor(varAssignement,dtype=float),axes=1)
        deltaE += J
    deltaE = tf.multiply(deltaE,tf.convert_to_tensor(multp_fac,dtype=float))
    #deltaE = tf.math.reduce_sum(deltaE,axis=0)
    #output: deltaE's
    return deltaE


def DigitalAnnealing3(pbf, steps, initialTemperature, seed=42):
    """
    Does a linear schedule of simulated annealing and returns a variable assignement and the energy value
    Input:  pbf: a dictionary like function: {(1,2,4): 3.4, (3,): 7, ...} <-> 3.4 * x_1x_2x_4 + 7x_3
            steps: number of iterations
            initialTemperature: higher energy -> higher probability to switch initially even if it does yield a better solution
    """
    offset_increase_rate = 0   #TODO offset dynamisch bestimmen?
    varAssignement = getInitialVarAssignement(pbf, seed)
    #convert to Tensor
    Tensor_list = ConvertPBFtoTensorForm(pbf,np.max(list(varAssignement.keys())))
    varAssignement_vek = ConvertDicToVektor(varAssignement)
    random.seed(seed)
    E_offset = 0

    for step in range(1, steps +1):
        T = step * initialTemperature
        eval =list(range(len(varAssignement)))
        #paralell part
        parallel = p_imap(lambda x: EvalDeltaEAsTensor(Tensor_list, varAssignement_vek,x), eval)
        deltaE=list(parallel)
        select = []
        if 1:
            for i in range(len(deltaE)):
                if deltaE[i]>0:
                    if allow_anyway(deltaE[i],T):
                        select.append(i)
                else:
                    select.append(i)
        
            #Choose random variable out of the "zulässigen" for the variable switch
            if select != []: 
                var = select[random.randint(0, len(select)-1)]
                if varAssignement[var] == 0:
                    varAssignement[var] = 1
                else:
                    varAssignement[var] = 0
                E_offset=0
            else:
                E_offset += offset_increase_rate 

    former_energy=evalPBF(pbf, varAssignement)

    return -former_energy, varAssignement



def DigitalAnnealing_QUBO(h,J,pbf, steps, initialTemperature, seed=42):
    """
    Does a linear schedule of simulated annealing and returns a variable assignement and the energy value
    Input:  pbf: a dictionary like function: {(1,2,4): 3.4, (3,): 7, ...} <-> 3.4 * x_1x_2x_4 + 7x_3
            steps: number of iterations
            initialTemperature: higher energy -> higher probability to switch initially even if it does yield a better solution
    """
    offset_increase_rate = 0 
    varAssignement = getInitialVarAssignement(pbf, seed)
    varAssignement_vek = ConvertDicToVektor(varAssignement)
    random.seed(seed)
    E_offset = 0

    for step in range(1, steps +1):
        T = step * initialTemperature
        if 1:
            deltaE = Eval_delta_E_as_QUBO(h,J,varAssignement_vek)
        select = []
        if 1:
            for i in range(len(deltaE)):
                if deltaE[i]>0:
                    if allow_anyway(deltaE[i],T):
                        select.append(i)
                else:
                    select.append(i)
        
            #Choose random variable out of the "zulässigen" for the variable switch
            if select != []: 
                var = select[random.randint(0, len(select)-1)]
                if varAssignement_vek[var] == 0:
                    varAssignement_vek[var] = 1
                else:
                    varAssignement_vek[var] = 0
                E_offset=0
            else:
                E_offset += offset_increase_rate 
    varAssignement = dict(enumerate(map(int, varAssignement_vek)))
    former_energy=evalPBF(pbf, varAssignement)

    return former_energy, varAssignement
