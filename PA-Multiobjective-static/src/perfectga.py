##########################################################################################
######                       Module Perfect Information - Genetic Algorithm                         ######
# this file compiles the methods of the symmetric (perfect) information case 
# of the principal-agent model approximation with evolutionary or genetic algorithms
##########################################################################################

from random import Random
from time import time
import inspyred
import math
import pandas as pd
import numpy as np
import os
from numpy import *
import uuid #unique id
import sys

#importing our .py's files
from src.params import *

# creating a database
# database of current simulation
name = "static_perf_10000_fix_cro_"+str(crossover)+"_mut_"+str(mutation)+"_temporal"
# name of columns
cols = ['sim','princ', 'agent', 'wage', 'effort', 'risk_averse','init_pop', 'gen','cutoff']
# dataframe with current simulation
db_sims = pd.DataFrame([], columns = cols ).set_index('sim') 


def my_observer_perf(population, num_generations, num_evaluations, args):
    """ This observer is called every each generation and prints important information. 
    It also saves the intermediate generations

    """
    arc = args['_ec'].archive
    pareto_set = []
    vU_ga, vV_ga, vW_ga, va_ga = [], [], [], []  
    
    if(num_generations%(max_gen//1000)) == 0:
        #Analysing Convergence
        
        for f in arc: 
            pareto_set.append([f.fitness[0],f.fitness[1]])
        
        #Evaluating HV Hypervolume Indicator
        global hypervolume_ind
        global generations_hv
        hypervolume_ind.append(inspyred.ec.analysis.hypervolume(pareto_set, reference_point=(0,0)))
        #hypervolume_ind_n.append(inspyred.ec.analysis.hypervolume(pareto_set, reference_point=None))
        generations_hv.append(num_generations)
        #print("******gen****** ::: ",num_generations," *****HV****** ::: ",hypervolume_ind[-1])
        #print("********HV**none**** ::: ",inspyred.ec.analysis.hypervolume(pareto_set, reference_point=None))
        
        
    if num_generations in gens_observations:#(num_generations%(max_gen//10)) == 0: 
        best = max(population)
        pareto_set = []
        print('***generation*** {0:6}  -- ***init Pop*** {1} : ***pop size*** {2} : ***num eval*** {3} '.format(num_generations, 
                                      len(population), len(arc), num_evaluations))
        #Saving intermediate generation:
        
        #final candidates(compensations) and final fitness(utilities)
         #extracting the compensations and utilities of the last generation
        for f in arc: 
            vU_ga.append(f.fitness[0]) #utility principal
            vV_ga.append(f.fitness[1]) #utility agent
            vW_ga.append(f.candidate[0]) #compensations 
            va_ga.append(f.candidate[1]) #effort
            pareto_set.append([f.fitness[0],f.fitness[1]])
        
        #tag of intermediate cutoff
        cutoff =num_generations# num_generations//(max_gen//10)
        #Saving database
        global db_sims, sim_key
        #print("----cutoff -----",cutoff)
        df = creating_df_perf(name, vU_ga, vV_ga, vW_ga, va_ga, max_gen, init_pop_size,cutoff,sim_key)
        
        df_1 = db_sims
        ## Merge, add the news rows, last simulation
        
        db_sims = pd.concat([df_1,df],sort = False, axis=0)
#         print(db_sims.head())
        #db_sims.to_csv('{}.csv'.format(name))
        
        #Analysing Convergence
        global hv_ind_obs
        hv_ind_obs.append(inspyred.ec.analysis.hypervolume(pareto_set, reference_point=(0,0)))

        print("******gen****** ::: ",num_generations," ;*****HV****** ::: ",hypervolume_ind[-1],
              ';***points in database*** :::',len(db_sims))
        
# this method replace the population from last generation with a new population 
def my_replacer_perf(random, population, parents, offspring, args):
    #print("Replacing ..... population : {} offprings {} parents {}".format(len(population), len(offspring),len(parents)))
    psize = len(population)
    #population.sort(reverse=True) # ordering the current population , n log n
    #this number matters!!! proportion of popoulation directly to survivors
    K = 10
    survivors = population[:psize // K] #selects the best len(population)/1000 : survivors
    num_remaining = psize - len(survivors) #remaining population = population - survivors
    for i in range(num_remaining): #then for each one of the remaining population you randomnly choice one offpring
        survivors.append(random.choice(offspring)) ### how many offprings are? how are they chosen? 2000
    return survivors #then you end up with the same number of population. , the best + some random offprings     

# the generator it is just for the first generation
# generates one random sample, one "person" of the population.  
def custom_generator_perf(random, args):
    global init_pop_random
    if init_pop_random:
        nr_inputs = args.get('nr_inputs', 1) #nr_input # columns of vector?
        rangeLow = [random.uniform(0.0,YH)] #generates random number from [0,Yl]
        #rangeHigh = [random.uniform(0.0,4*YH)] #generates random number from [0,YH]
        rangeHigh = [random.uniform(0.0,YL)] # effort is always smaller than 2 (from simulations) 
        individual = rangeLow + rangeHigh  #return the list that  results from the concatenation of the two previous generated lists.  
    else:
        #################### seeding ##########################
        global init_pop_seed
        global count_seed
        individual = init_pop_seed[count_seed]
        count_seed += 1
        #print("Individual seeded ", individual)
    return individual

#it receives w and a, and checks that w in [0,YH] and that variables are no negative
def validate_limit_perf(comps):  #Verificación de los perfiles de salarios.
    # you need to check both boundaries in both compensations ##################
    
    #if compensation or effort are negative , then make it 0
    if comps[0]<0:
        comps[0] = limit 
    if comps[1]<0:
        comps[1] = limit
    #if compensation > YH , then make it YH
    if comps[0]>YH:
        comps[0]=YH
    return comps

## Principal and Agent utility, U and V, 

#Return the principal's utility, risk neutral, given effort a and present compensations W
def principal_utility_perf(W,a):   
    U = (1 - math.exp(- a)) * YH + math.exp(- a) * YL - W;
    return U

#Return the agent's utility, risk averse, given effort a and present compensations w, W and future compensation
def agent_utility_perf(W,a): 
    V = (W**(1 - h)) / (1 - h) - a;
    return V 

#The evaluator choose from the candidates, in every generation
def custom_evaluator_perf(candidates, args):
    #simulation key 
    fitness = [] 
    # this is alwayw 1 in the static case
    contract = int(n_inputs / 2)
    # loops over each candidate (wage, effort)
    for c in candidates:
        c = validate_limit_perf(c) #check for compensation and effort bounds
       
        #Principal Utility given contract(wage,action)
        U = principal_utility_perf(c[0],c[1]) 
        #Agent Utility given contract(wage,action)
        V = agent_utility_perf(c[0],c[1]) 

        ## Drop off if the principal and agent utilities are not feasible; U < 0 
        if U < 0 or V < 0: 
            U = -100
            V = -100
            
        fitness.append(inspyred.ec.emo.Pareto([U, V])) #only append the Pareto Optimal solution .V: U_agent ,U: U_principal 
        #A Pareto solution is a set of multiobjective values that can be compared to other Pareto values using Pareto preference.
        #This means that a solution dominates, or is better than, another solution
        #if it is better than or equal to the other solution in all objectives and strictly better in at least one objective.
    return fitness  

############ Database Method ###########
#This method creates a new  dataframe for the last simulation 
def creating_df_perf(name, vU, vV, vw, va, max_gen, init_pop_size,cutoff,sim_key=uuid.uuid4()):
    # intialize data of lists. 
    leng = len(vU)
    data = {'sim':[str(sim_key)]*leng, 'princ':vU, 'agent':vV, 'wage':vw, 
            'effort':va, 'risk_averse': [h]*leng,
            'init_pop':[init_pop_size]*leng, 'gen':[max_gen]*leng ,'cutoff':[cutoff]*leng} 
    # Create DataFrame 
    df = pd.DataFrame(data).set_index('sim')

    return df

##############  Simulation ###############


# .
def simulate_perf():
    """This method runs a simulation with a new sim_key and current global parameters

    Returns:
        dataframe: Data of the simulation. 
        list of arguments: The data of the last simulation in order is principal utilities, agent utilities, wages, efforts, init_pop_size, max_gen
        dataframe: Data of the hypervolume dataframe. 
    """
    global db_sims, sim_key
    sim_key=uuid.uuid4()
    #Reinitiation of variables , empty lists
    #va_ga, vw_ga, vU_ga, vV_ga = [], [], [], []

    prng = Random()
    prng.seed(time())
    ea = inspyred.ec.emo.NSGA2(prng)
    ea.variator = [inspyred.ec.variators.blend_crossover,
                   inspyred.ec.variators.gaussian_mutation]
    ea.terminator = inspyred.ec.terminators.generation_termination
    #dimensions=int(3*n_inputs / 2)
    ea.observer = [my_observer_perf]
    ea.replacer = my_replacer_perf
    
    
    # inspyred library evolves initial population
    final_pop = ea.evolve(generator=custom_generator_perf,
                          #evaluator=inspyred.ec.evaluators.parallel_evaluation_mp, #The function assigns the evaluation of each candidate (=pop_size) to its own job
                          #mp_evaluator=CustomEvaluator,
                          #mp_nprocs=4,  
                          evaluator=custom_evaluator_perf,
                          pop_size=init_pop_size,
                          maximize = True,
                          r_inputs = n_inputs,
                          bounder=inspyred.ec.Bounder(None, None),
                          #bounder=inspyred.ec.Bounder([0.0]*dimensions, [1.0]*dimensions),
                          max_generations = max_gen,
                          crossover_rate = crossover,
                          mutation_rate = mutation)

    #final candidates(compensations) and final fitness(utilities)
    vU_ga, vV_ga, vW_ga, va_ga = [], [], [], []  
    if display:
        final_arc = ea.archive

        #with the purpose of ploting, extracting the compensations and utilities of final generation
        for f in final_arc: 
            vU_ga.append(f.fitness[0]) #utility principal
            vV_ga.append(f.fitness[1]) #utility agent
            vW_ga.append(f.candidate[0]) #compensations 
            va_ga.append(f.candidate[1]) #effort


    print('----------population in final arc {0}, with h = {1}----------'.format(len(vU_ga),h))
    #pd.DataFrame({'x':x,'y':y}).to_csv('resultados_5k_h097.csv')
    #pd.DataFrame(final_arc).to_csv('acciones_5k_h097.csv')
    

    # saving database with cuts
    db_sims.to_csv('./data/{}.csv'.format(name+str(sim_key)[:7]))
    
    #SAVING HYPERVOLUME
    df_hyp = pd.DataFrame({'gen': generations_hv,'indicator':hypervolume_ind,'sim':[sim_key]*len(generations_hv)})
    pd.DataFrame(df_hyp).to_csv('./data/hv_{}.csv'.format(name+str(sim_key)[:7]))
    
    print("archive name: ", name+str(sim_key)[:7])   
    print("Currently in database \'simulations\' from Risk Averse = ", db_sims['risk_averse'].unique())
    print("--------------------Ended simulation-------------------")
    return db_sims,[vU_ga,vV_ga,vW_ga,va_ga,init_pop_size,max_gen],df_hyp

##################### seeding ##########################

# This method clear all variables and generates a new dataframe in which archives 
# of new simulations will be
def reinit_variables_perf(cara=0.5):
        global name, db_sims, hypervolume_ind, hypervolume_ind_n
        global hv_ind_obs,generations_hv, gens_observations,gens_obs_percent
        # risk aversion
        h = cara
        # calculating cutoff generations
        gens_observations = [int(max_gen*i) for i in gens_obs_percent ]
        name = "static_perf_"+str(max_gen)+"_h"+str(h)+"_cro_"+str(crossover)+"_mut_"+str(mutation)+"_key_"
        print('archive params :', name)
        #cols = ['sim','vU', 'vV', 'vw', 'va', 'RiskAverse',
        #    'init_pop', 'gen','cutoff']
        cols = ['sim','princ', 'agent', 'wage', 'effort', 'risk_averse',
            'init_pop', 'gen','cutoff']
        db_sims = pd.DataFrame([], columns = cols ).set_index('sim') 
        hypervolume_ind,generations_hv = [],[]
        hypervolume_ind_n, hv_ind_obs = [],[]
        

# Sets seeding mode and verify that the initial seed is adequate
def reset_seeding(seeds):
        # Setting up seed
    global init_pop_random, count_seed, init_pop_seed, init_pop_size
    if not init_pop_random:
        #init_pop_seed = [w_l[i] + w_h[i] for i in range(len(w_l))] #here the population seed must be set, taking last simulation as seed
        init_pop_seed = seeds
        count_seed = 0
        print("********  Seeding mode enable ********")
    else:
        print("**** Random population mode enable ****")

    #making initial population size equal than the population seed size
    if not init_pop_random:# and (len(init_pop_seed)< init_pop_size):
        init_pop_size = len(init_pop_seed)
        if init_pop_size == 0:
            init_pop_random = True
            print("Seeding is not possible, changing mode to random population")