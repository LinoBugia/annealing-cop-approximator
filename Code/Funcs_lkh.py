# This code includes mofified functions of the repo https://github.com/kikocastroneto/lk_heuristic.git 
# and was used to solve the Traveling Salesman problem.


from lk_heuristic.models.tsp import Tsp
from lk_heuristic.utils.cost_funcs import euc_2d
from lk_heuristic.models.node import Node2D

import math
import logging
import random
from itertools import permutations
from Funcs_TSP import *
import time

def solve2(shuffle,tsp_file=None, coords=None,solution_method=None, runs=1, backtracking=(5, 5), reduction_level=4, reduction_cycle=4, tour_type="cycle", file_name=None, logging_level=logging.DEBUG):
    """
    Solve a specific tsp problem a certain amount of times using the tsp_file input and the desired solution method. If this functions is called with no supplied inputs, the interactive inputs will be collected through the terminal. The best solution is parsed to .tsp file and exported to solution folder. 

    :param tsp_file: the .tsp file to be solved
    :type tsp_file: str
    :param solution_method: the method to be used when solving the tsp
    :type solution_method: function
    :param runs: the number of improve runs to be performed
    :type runs: int
    :param backtracking: the number of closest neighbors on each search level
    :type backtracking: tuple
    :param reduction_level: the search level where reduction starts being applied
    :type reduction_level: int
    :param reduction_cycle: the search cycle where reduction starts being applied
    :type reduction_cycle: int
    :param tour_type: the type of the tour (either 'path' or 'cycle')
    :type tour_type: str
    :param file_name: the name of the tsp file to be written
    :type file_name: str
    :param logging_level: the verbosity level for more or less details during execution
    :type logging_level: int
    """

    # get interactive inputs if input is not supplied at function
    #if not (tsp_file or solution_method):
    #    tsp_file, solution_method = get_interactive_inputs()

    # setup the logger
    logging.basicConfig(level=logging_level, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    logger = logging.getLogger(__name__)

    # parse the .tsp file
    #logger.info(f"Importing .tsp file '{os.path.basename(tsp_file)}'")
    tsp_header = "blank"
    tsp_nodes = coords

    # get the cost function

    #logger.info(f"Using '{cost_function.__name__}' for edge type '{tsp_header['EDGE_WEIGHT_TYPE']}'")

    # setup some information of the run
    best_tour = None  # the best tsp tour nodes found so far
    best_cost = math.inf  # the best cost found so far
    mean_cost = 0  # the mean cost value of all runs

    # create the initial tsp instance
    logger.info("Creating TSP instance")
    tsp = Tsp(tsp_nodes, euc_2d, shuffle=shuffle, backtracking=backtracking, reduction_level=reduction_level, reduction_cycle=reduction_cycle, tour_type=tour_type, logging_level=logging_level)

    # looping through each run
    logger.info("Starting improve loop")
    for run in range(1, runs + 1):

        # shuffle the tour nodes
        if tsp.shuffle:
            tsp.tour.shuffle()

        # initialize tour cost
        tsp.tour.set_cost(tsp.cost_matrix)

        # execute the improvement method and timeit
        start_time = time.time()
        tsp.methods[solution_method]()
        end_time = time.time()

        # update best solution found so far (if current solution is the best)
        if tsp.tour.cost < best_cost:
            best_tour = tsp.tour.get_nodes()
            best_cost = tsp.tour.cost

        # update mean cost value
        mean_cost += (tsp.tour.cost - mean_cost) / run

        # log the information of current run
        logger.info(f"[Run:{run}] --> Cost: {tsp.tour.cost:.3f} / Best: {best_cost:.3f} / Mean: {mean_cost:.3f} ({end_time - start_time:.3f}s)")
    return best_tour,best_cost
    

def solve_lkh(coords,runs=1,visualize =False,shuffle=False):

    shuffle=False 
    backtracking=(5, 5) 
    reduction_level=4
    reduction_cycle=4
    tour_type="cycle"
    logging_level=logging.INFO

    coords2=[]
    for coord in coords:
        coords2.append(Node2D(float(coord[0]), float(coord[1])))
    start_time=time.time()
    best_tour,best_cost = solve2(coords=coords2,runs=runs,solution_method="lk1_improve",shuffle=shuffle)
    #print(best_tour)
    print(str(time.time()-start_time) + " sec")
    #Convert to different format
    solution=[]
    for i in range(len(best_tour)):
        x=best_tour[i].x
        y=best_tour[i].y
        solution.append([x,y])
    if visualize:
        plot_tour(solution)
    return solution,best_cost
# setup the logger
if 0:
    coords = make_random_coords(0,1,0,1,100,seed = 42)
    solve_lkh(coords,visualize=True)