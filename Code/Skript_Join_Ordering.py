# NOTE: This script requires ProblemGenerator.py and QUBOGenerator.py from
# https://github.com/lfd/vldb24 (GPL-3.0). They are not included in this
# repository — download them from the upstream repo into Code/ before running.

import numpy as np

from QUBOGenerator import *
from ProblemGenerator import *

from Funcs_Optimizers import *
import gzip
#problem_path = r"C:\Github\fujitsuclone\Problems\benchmarks\sqlite\q109"                        # windows
#problem_path = r"/home/lino/Dokumente/GitHub/fujitsuclone/Problems/benchmarks/sqlite/q109"      # linux     # mac
from Funcs_pbfGenerators import convert_IBMQ_QUBO_to_pbf


def load_compressed_data(path, filename):
    filepath = os.path.join(path, filename)
    if not os.path.exists(filepath):
        return None
    with gzip.open(filepath, mode='rt', encoding='UTF-8') as f:
        return json.load(f)

problem_path = "/Users/lino/Documents/python/fujitsuclone/Problems/benchmarks/sqlite/q200"
card, pred,pred_sel = get_join_ordering_problem(problem_path)
path = "/Users/lino/Documents/python/vldb24/base/Experiments/Data_saved/da/benchmarks/sqlite/q200/legacy_approximation/thres_config_4/penalty_scaling_2/annealing/1000000_iterations/100_shots/sample_0/"

if 1:
    IBMQ_QUBO = generate_IBMQ_QUBO_for_left_deep_trees(card, pred, pred_sel, log_thres  = 0.63, num_decimal_pos=2, penalty_scaling=1, minimum_penalty_weight=20)
    test = 0
    pbf,variables = convert_IBMQ_QUBO_to_pbf(IBMQ_QUBO)
    pbf_var_dict = createPolyDict(pbf,variables)


    response = load_compressed_data(path, "response.txt")
    Energies=[]
    for i in range(len(response[0])):
        Energies.append(evalPBF_List(pbf,response[0][i][0]))



    results_fujitsu = 0

    #pbf2 = createPoly(20,2,0.8)
    num_MC = 15
    steps=8000

    degree = 2

    type_alg = "SimulatedAnnealing"

    Offset_increase = 100

    #cooling_param = ["linear", 1000,1000000]
    cooling_param = ["logarithmic", 70,0]
    #cooling_param = ["exponential", 1000000,1000000]
    seed_gen = 76243
    seed_rand =[]
    random.seed(seed_gen)
    for i in range(50):
        seed_rand.append(random.uniform(0,1000))



    #print("feature generation time:" + str(time.time() - start_time_gen))
    pbf_min_solver(pbf,pbf_var_dict,"simulatedAnnealing",10000,50,cooling_param,seed_rand,seed_gen,visual_inst=True,save_csv=False)
    #pbf_min_solver(pbf,pbf_var_dict,"scaAnnealing",200,1,cooling_param,[seed_rand[0]],seed_gen,visual_inst=True,save_csv=True)
    #pbf_min_solver(pbf,pbf_var_dict,"digitalAnnealing_parallel",steps,num_MC,cooling_param,seed_rand,seed_gen,visual_inst=True,offset_increase_rate=Offset_increase,save_csv=False)
    pbf_min_solver(pbf,pbf_var_dict,"digitalAnnealing",steps,num_MC,cooling_param,seed_rand,seed_gen,visual_inst=True,offset_increase_rate=Offset_increase,save_csv=False)
    print("Energies fujitsu:")

    print(Energies)
    #
