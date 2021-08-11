##########################################################################################
######                       Module Imperfect Information - Genetic Algorithm                  ######
# this file compiles the methods of the assymmetric (imperfect) information case 
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

# creating or redefining a database
# database of current simulation
name = "static_perf_10000_fix_cro_"+str(crossover)+"_mut_"+str(mutation)+"_temporal"
# name of columns
cols = ['sim','princ', 'agent', 'wl', 'wh', 'risk_averse','init_pop', 'gen','cutoff']
# dataframe with current simulation
db_sims = pd.DataFrame([], columns = cols ).set_index('sim') 



def my_observer_imp(population, num_generations, num_evaluations, args):
    """ This observer is called once each generation and prints important information. 
    It also saves the intermediate generations

    """
    archive = args['_ec'].archive
    pareto_set = []
    if(num_generations%(max_gen//1000)) == 0:
        #Analysing Convergence
        final_arc = archive
        for f in final_arc: 
            pareto_set.append([f.fitness[0],f.fitness[1]])
        
        #Evaluating HV Hypervolume Indicator
        global hypervolume_ind
        global generations_hv
        hypervolume_ind.append(inspyred.ec.analysis.hypervolume(pareto_set, reference_point=(0,0)))
        #hypervolume_ind_n.append(inspyred.ec.analysis.hypervolume(pareto_set, reference_point=None))
        generations_hv.append(num_generations)
#         print("******gen****** ::: ",num_generations," *****HV****** ::: ",hypervolume_ind[-1])
        #print("********HV**none**** ::: ",inspyred.ec.analysis.hypervolume(pareto_set, reference_point=None))
        
    if num_generations in gens_observations:#(num_generations%(max_gen//10)) == 0: 
        best = max(population)
        pareto_set = []
        print('***generation*** {0:6}  -- ***init Pop*** {1} : ***pop size*** {2} : ***num eval*** {3} '.format(num_generations, 
                                      len(population), len(archive), num_evaluations))
        #Saving intermediate generation:
        
        #final candidates(compensations) and final fitness(utilities)
        vU_ga, vV_ga, wh_ga, wl_ga = [], [], [], []  
        final_arc = archive
        
         #extracting the compensations and utilities of the last generation
        for f in final_arc: 
            vU_ga.append(f.fitness[0]) #utility principal
            vV_ga.append(f.fitness[1]) #utility agent
            wl_ga.append(f.candidate[0]) #compensation low
            wh_ga.append(f.candidate[1]) #compensation high
            pareto_set.append([f.fitness[0],f.fitness[1]])
        
        #tag of intermediate cutoff
        cutoff = num_generations# num_generations//(max_gen//10)
        #Saving database
        global db_sims, sim_key
        #print("----cutoff -----",cutoff)
        df = creating_df_imp(name, vU_ga, vV_ga,wl_ga, wh_ga, max_gen,
                         init_pop_size,cutoff,sim_key)
        #print(db_sims.head())
        df_1 = db_sims
        ## Merge, add the news rows, last simulation
        db_sims = pd.concat([df_1,df],sort = False, axis=0)
        
        #Analysing Convergence
        global hv_ind_obs
        hv_ind_obs.append(inspyred.ec.analysis.hypervolume(pareto_set, reference_point=(0,0)))

        #db_sims.to_csv('{}.csv'.format(name))
        
        print("******gen****** ::: ",num_generations," ;*****HV****** ::: ",hypervolume_ind[-1],';***points in database*** :::',len(db_sims))
        
   
# this method replace the population from last generation with a new population 
def my_replacer_imp(random, population, parents, offspring, args):
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

#the generator it is just for the first generation, the one you start with. ???
# generates one random sample, one "person" of the population.  
def custom_generator_imp(random, args):
    global init_pop_random
    if init_pop_random:
        nr_inputs = args.get('nr_inputs', 1) #nr_input # columns of vector?
        rangeLow = [random.uniform(0.0,YL)] #generates random number from [0,Yl]
        #rangeHigh = [random.uniform(0.0,4*YH)] #generates random number from [0,YH]
        rangeHigh = [random.uniform(0.0,YH)] 
        #Swaping to mantain compensation_H > compesation_L
        if rangeHigh[0] < rangeLow[0]: #if the generated with the lower higher bound(yl) is greater than the generated with yH
            rangeHigh[0],rangeLow[0] = rangeLow[0],rangeHigh[0]  #then swap #swaping works great!
        individual = rangeLow + rangeHigh  #return the list that  results from the concatenation of the two previous generated lists.  
    else:
        #################### seeding ##########################
        global init_pop_seed
        global count_seed
        individual = init_pop_seed[count_seed]
        count_seed += 1
        #print("Individual seeded ", individual)
    return individual

#it receives w_l, w_h and checks that w_l in [0,YL] and w_h in [0,YH]
def validate_limit_imp(comps):  #Verificación de los perfiles de salarios.
    # you need to check both boundaries in both compensations ##################

    #if low/high compensation is negative , then make it 0
    if comps[0]<0:
        comps[0] = limit 
    if comps[1]<0:
        comps[1] = limit
    #if low compensation > YL , then make it Yl
    if comps[0]>YL:
        comps[0]=YL
     #if high compensation > YH , then make it YH
    if comps[1]>YH:
        comps[1]=YH
    return comps
    
    

# Finds the best possible action for the agent given the contract. it has to be positive and defined
# what if a <1, it should be better to tossed this solution?? ERROR? 
def validate_action_imp(wl,wh):
    a = wh**(1 - h)/(1 - h) - wl**(1 - h)/(1 - h);
    if a < 1:
        return 0, True
    else: return math.log(a), False   
    
## U and V, Principal and Agent  utility
#Return the principal's utility, risk neutral, given effort a and present compensations w, W
def principal_utility_imp(wl,wh,a):   
    U = (1 - math.exp(- a)) * (YH - wh) + math.exp(- a) * (YL- wl);
    return U

#Return the agent's utility, risk averse, given effort a and present compensations w, W and future compensation
def agent_utility_imp(wl,wh,a): 
    V = (1 - math.exp(- a)) * (wh**(1 - h) / (1 - h) - a) + math.exp(- a) * (wl**(1 - h) / (1 - h) - a);
    return V 
############

#The evaluator choose from the candidates, in every generation
def custom_evaluator_imp(candidates, args):
    fitness = [] 
    contract = int(n_inputs / 2)
    for c in candidates:
        # c has low and high compensation [w,W] 
        c = validate_limit_imp(c) #check for low compensation first
        # Las siguientes variables se irán redefiniendo conforme se vaya evaluando el algoritmo.
       

        # a = the effort related with these compensations
        a, indef = validate_action_imp(c[0],c[1]) # "best" action given W,w,VH,VL            

        U = principal_utility_imp(c[0],c[1],a) #Principal Utility given contract(W,W,action,UH,UL )

        V = agent_utility_imp(c[0],c[1],a) 

        ## Drop off if the effort is undefined #a==-1 and when W<w
        if c[1] < c[0] or indef : 
            U = -100
            V = -100


        fitness.append(inspyred.ec.emo.Pareto([U, V])) #only append the Pareto Optimal solution .V: U_agent ,U: U_principal 
    #A Pareto solution is a set of multiobjective values that can be compared to other Pareto values using Pareto preference.
    #This means that a solution dominates, or is better than, another solution
    #if it is better than or equal to the other solution in all objectives and strictly better in at least one objective.
    return fitness  



####  Database method #####
#This method Creates a new  dataframe for the last simulation 


def creating_df_imp(name, vU, vV, wl, wh, max_gen, init_pop_size,cutoff,sim_key=uuid.uuid4()):
    # intialise data of lists. 
    leng = len(vU)
    data = {'sim':[sim_key]*leng, 'princ':vU, 'agent':vV, 'wl':wl, 
            'wh':wh, 'risk_averse': [h]*leng,
            'init_pop':[init_pop_size]*leng, 'gen':[max_gen]*leng ,'cutoff':[cutoff]*leng} 
    # Create DataFrame 
    df = pd.DataFrame(data).set_index('sim')

    return df

##############  Simulation  ###############
#number of generations and initial population
# calculating cutoff generations
gens_observations = [int(max_gen*i) for i in gens_obs_percent ]

def simulate_imp():
    """This method runs a simulation with a new sim_key and current global parameters

    Returns:
        dataframe: Data of the simulation. 
        list of arguments: The data of the last simulation in order is principal utilities, agent utilities, wages, efforts, init_pop_size, max_gen
        dataframe: Data of the hypervolume dataframe. 
    """
    global db_sims, sim_key
    #generating new key
    sim_key=uuid.uuid4()

    prng = Random()
    prng.seed(time())
    ea = inspyred.ec.emo.NSGA2(prng)
    ea.variator = [inspyred.ec.variators.blend_crossover,
                   inspyred.ec.variators.gaussian_mutation]
    ea.terminator = inspyred.ec.terminators.generation_termination
    #dimensions=int(3*n_inputs / 2)
    ea.observer = [my_observer_imp]
    ea.replacer = my_replacer_imp

    # genetic algorithm evolution
    final_pop = ea.evolve(generator=custom_generator_imp,
                          #evaluator=inspyred.ec.evaluators.parallel_evaluation_mp, #The function assigns the evaluation of each candidate (=pop_size) to its own job
                          #mp_evaluator=CustomEvaluator,
                          #mp_nprocs=4,  
                          evaluator=custom_evaluator_imp,
                          pop_size=init_pop_size,
                          maximize=True,
                          r_inputs = n_inputs,
                          bounder=inspyred.ec.Bounder(None, None),
                          #bounder=inspyred.ec.Bounder([0.0]*dimensions, [1.0]*dimensions),
                          max_generations = max_gen,
                          crossover_rate = crossover,
                          mutation_rate = mutation)

    #final candidates(compensations) and final fitness(utilities)
    vU, vV, wl, wh = [], [], [], []  
    if display:
        final_arc = ea.archive

        #this loop extracts the compensations and utilities of final generation
        for f in final_arc: 
            vU.append(f.fitness[0]) #utility principal
            vV.append(f.fitness[1]) #utility agent
            wl.append(f.candidate[0]) #Low compensations 
            wh.append(f.candidate[1]) #Hight compensations

    print('----------population in final arc {0}, with h = {1}----------'.format(len(vU),h))
    

    # saving final database with cuts
    db_sims.to_csv('./data/{}.csv'.format(name+str(sim_key)[:7]))
    #Saving Hypervolume Indicators
    df_hyp = pd.DataFrame({'gen': generations_hv,'indicator':hypervolume_ind,'sim':[sim_key]*len(generations_hv)})
    pd.DataFrame(df_hyp).to_csv('./data/hv_{}.csv'.format(name+str(sim_key)[:7]))
    
        
    print("archive name: ", name+str(sim_key)[:7])  
    print("Currently in database \'simulations\' from Risk Averse = ", db_sims['risk_averse'].unique())
    print("--------------------Ended simulation-------------------")
    return db_sims,[vU,vV,wl,wh,init_pop_size,max_gen],df_hyp


##################### seeding ##########################

# This method clear all variables and generates a new data frame 
# in which archives of new simulations will be
def reinit_variables_imp(cara = 0.5):
    global name, db_sims, hypervolume_ind, hypervolume_ind_n
    global hv_ind_obs,generations_hv, gens_observations,gens_obs_percent
    h = cara
    # calculating cutoff generations
    gens_observations = [int(max_gen*i) for i in gens_obs_percent ]
    name = "static_imp_"+str(max_gen)+"_h"+str(h)+"_cro_"+str(crossover)+"_mut_"+str(mutation)+"_key_"
    print('archive params :', name)
    #cols = ['sim','vU', 'vV', 'wl', 'wh', 'RiskAverse',
    #    'init_pop', 'gen','cutoff']
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