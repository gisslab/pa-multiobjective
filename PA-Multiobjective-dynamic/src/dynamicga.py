##########################################################################################
######       Module Dynamic Imperfect Information - Genetic Algorithm                         ######
# this file compiles the methods of the asymmetric (imperfect) information case 
# of the dynamic principal-agent model approximation with evolutionary or genetic algorithms
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
name = "dynamic_perf_10000_fix_cro_"+str(crossover)+"_mut_"+str(mutation)+"_temporal"
# name of columns
cols = ['sim','u_principal', 'u_agent', 'w_l', 'w_h', 'gen']
# dataframe with current simulation
df_sim = pd.DataFrame([], columns = cols ).set_index('sim') 

#my observer it is call one's every each generation and prints important information
def my_observer(population, num_generations, num_evaluations, args):
    """ This observer is called once each generation and prints important information. 
    It also saves the intermediate generations

    """
    #Analysing Convergence
    archive = args['_ec'].archive
    pareto_set = []

       #Hypervolumes, Convergence Analysis 
    if (num_generations % (max_gen//100)) == 0:
        
        final_arc = archive
        
        for f in final_arc: 
            pareto_set.append([f.fitness[0],f.fitness[1]])
        
        #Evaluating HV Hypervolume Indicator
        global hypervolume_ind
        global generations_hv
        hypervolume_ind.append(inspyred.ec.analysis.hypervolume(pareto_set, reference_point=(0,0)))#, reference_point=None)
        generations_hv.append(num_generations)
        #print("********HV*********** ::: ",hypervolume_ind[-1])  
   
    if num_generations in gens_observations:

        print('******gen****** ::: {0:6}  -- ***init Pop*** {1} : ***pop size*** {2} : ***num eval*** {3} '.format(num_generations, 
                                      len(population), len(archive), num_evaluations))
        #Saving intermediate generation:
        
        #final candidates(compensations) and final fitness(utilities)
        vU_ga, vV_ga, vW_ga, vw_ga = [], [], [], [] 
        archive = args['_ec'].archive
        final_arc = archive

        #extracting the compensations and utilities of the last generation
        for f in final_arc: 
            vU_ga.append(f.fitness[0]) #utility principal
            vV_ga.append(f.fitness[1]) #utility agent
            vW_ga.append(f.candidate[0:contract_len]) #low compensations 
            vw_ga.append(f.candidate[contract_len:n_inputs]) #high compensation
            
        #tag of intermediate cutoff
        cutoff = num_generations
        #Saving database
        global df_sim

        #change this aggregate cutoffs
        df = creating_df(vU_ga, vV_ga, vW_ga,vw_ga, cutoff)
        
        ## Merge, add the news rows, last simulation
        df_1 = df_sim
        df_sim = pd.concat([df_1,df],sort = False, axis=0)

        #Analysing Convergence
        global hv_ind_obs
        hv_ind_obs.append(inspyred.ec.analysis.hypervolume(pareto_set, reference_point=(0,0)))


        #df_sim.to_csv('./data/{}.csv'.format(name))
        print("******gen****** ::: ",num_generations," ;*****HV****** ::: ",hypervolume_ind[-1],';***points in database*** :::',len(df_sim))
        
        

# this method replace the population from last generation with a new population 
def my_replacer(random, population, parents, offspring, args):
   # print("Replacing ..... population : {} offprings: {} parents: {}".format(len(population), len(offspring),len(parents)))
    psize = len(population)
    population.sort(reverse=True) # ordering the current population , n log n
    #this number matters!!! proportion of popoulation directly to survivors
    K = 10
    survivors = population[:psize // K] #selects the best len(population)/K : survivors
    num_remaining = psize - len(survivors) #remaining population = population - survivors
    for i in range(num_remaining): #then for each one of the remaining population you randomnly choice one offpring
        survivors.append(random.choice(offspring)) ### how many offprings are? how are they chosen? 2000
    return survivors #then you end up with the same number of population. , the best + some random offprings     

#the generator it is just for the first generation, the one you start with. ???
# generates one random sample, one "person" of the population.  
def custom_generator(random, args):
    global init_pop_random
    if init_pop_random:
        #################### generating ##########################
        nr_inputs = args.get('nr_inputs', 1) #nr_input # columns of vector
        rangeLow = [random.uniform(0.0,YL) for _ in range(int(n_inputs / 2))] #generates random number from [0,YL]
        rangeHigh = [random.uniform(0.0,YH) for _ in range(int(n_inputs / 2))] #generates random number from [0,YH]

        #for loop swaps compensations if necessary
        #the reason it swaps is to mantain compensation_H > compesation_L 
        for i in range(int(n_inputs / 2)): 
            if rangeHigh[i] < rangeLow[i]: #if the element generated with the lower higher bound(yl) is greater than the element generated with yH
                rangeLow[i],rangeHigh[i] =rangeHigh[i], rangeLow[i] # then swap
                #print('swaping L and H',rangeLow[i]," and ",rangeHigh[i])         
        #one individual is a list that  results from the concatenation of the two previous generated lists.  
        individual = rangeLow + rangeHigh 
        #print("Individual initialized ", individual)
    
    else:
        #################### seeding ##########################
        global init_pop_seed
        global count_seed
        individual = init_pop_seed[count_seed]
        count_seed += 1
        #print("Individual seeded ", individual)
    return individual


#it receives w, W and checks that w in [0,YL] and W in [0,YH]
def validate_limit(array, initial, final):  
    ######## , you need to check both boundaries in both compensations ##################
    # print("limits , final : ", final," initial: ",initial, "W :",len(array), "w[0] ", array[0])
    for i in range(initial, final): #just one time if you have one pair, for all w ,or W
        #if low/high compensation is negative , then make it 0
            
        if array[i] < 0: 
            array[i] = limit     
        #if high compensation > YH , then make it YH    
        elif array[i] > YH and initial > 0:    
            array[i] = YH           
        #if low compensation > YL , then make it Yl
        elif array[i] > YL and initial == 0: array[i] = YL            
    return array #it returns the new compensations between the bounds

# Finds the best possible action for the agent given the contract. it has to be positive and defined
# what if a <1, it should be better to tossed this solution?? ERROR? 
def validate_accion(W,w,VH,VL):
    a = (W**(1 - h))/(1 - h) - (w**(1 - h))/(1 - h) + beta * (VH - VL);
    if a < 1: #because effort = ln(a) must be positive
        return 0, True
    else: return math.log(a), False   
    

# Highest possible expected utility that the principal could get given a contract(W,w,UH,UL)
# calculate the utility of the principal  # todo unify
def high_principal_utility(W):
    UH = (YH - W);
    return UH

# calculate the utility of the principal  # todo unify
def low_principal_utility(w):      
    UL =(YL - w);
    return UL


# calculate the utility of the agent  # todo unify
def high_agent_utility(W,a):
    VH = ((W**(1-h))/(1-h)-a);
    return VH      

# calculate the utility of the agent  # todo unify
def low_agent_utility(w,a): 
    VL = ((w**(1-h))/(1-h)-a);
    return VL      
    
#Return the principal's utility, risk neutral, given effort a and present compensations w, W
def principal_utility(W,w,a,UH,UL):   
    U = (1 - math.exp(- a)) * (YH - W + beta * UH) + math.exp(- a) * (YL- w + beta * UL);
    return U

#Return the agent's utility, risk averse, given effort a and present compensations w, W and future compensation
def agent_utility(W,w,a,VH,VL): 
    V = (1 - math.exp(- a)) * (W**(1 - h) / (1 - h) - a + beta * VH) + math.exp(- a) * (w**(1 - h) / (1 - h) - a + beta * VL);
    return V 


#The evaluator choose from the candidates, in every generation
def custom_evaluator(candidates, args):
    fitness = [] 
    # clear list of eforts to only have population of efforts of last generation

    for W in candidates:
        # W = has low and high compensation [w,W] , starting with [w_n, w_{n-1},...w_1, W_n,...W_1]
        W = validate_limit(W, 0, contract_len) #check for low compensation(s) first
        W = validate_limit(W, contract_len, n_inputs) #check for high compensation, if contract= 1, n_inputs = 2
        #u_i = Agent Low and High utility vectors, length n+1 where n is amount of periods, starting with last period
        u_L,u_H = [0], [0]
        #v_i = Agent Low and High utility vectors, length n+1 where n is amount of periods, starting with last period
        v_L,v_H = [0], [0]
        #a = Agent action vectors, length n where n is amount of periods, starting with last period
        action,V_periods,U_periods = [],[],[]
        dominated = False
        for i in range(contract_len):  #in the case of n_inputs = 2 , it just happens one time
            # i = it refers to period n - i
            #indef = tell us if the contract is feasible
            #a = optimal action in period i (n - i)
            a, indef = validate_accion(W[contract_len+i],W[i],v_H[i],v_L[i]) # "best" action given W,w,VH,VL
            action.append(a)
            
            # U = expected utility in period i (n - i)
            U = principal_utility(W[contract_len+i],W[i],a,u_H[i],u_L[i]) #Principal Utility given contract(W,W,action,UH,UL) )
            U_periods.append(U) #add best action to list of principals expected utilities 
            # V = Agent expected utility in period i
            V = agent_utility(W[contract_len+i],W[i],a,v_H[i],v_L[i]) 
            V_periods.append(V) #add best action to list of principals expected utilities 
            
            # VH , VL= maximal and minimal agent discounted expected utility in period i
            VH = high_agent_utility(W[contract_len+i],a) #highest possible agents's expected utility
            v_H.append(VH)
            VL = low_agent_utility(W[i],a) #lowest possible agents's expected utility
            v_L.append(VL)
            # UH , UL = maximal and minimal principal discounted expected utility in period i
            UH = high_principal_utility(W[contract_len+i])
            u_H.append(UH)
            UL = low_principal_utility(W[i])
            u_L.append(UL) 

            ## Drop off if the effort is undefined and when W<w
            if W[contract_len+i]<W[i] or indef : 
                dominated = True
                #print('W: {0}   <   w: {1} => -100'.format(W[contract+i] ,W[i]))
        if dominated:        
            U = -100
            V = -100

        fitness.append(inspyred.ec.emo.Pareto([U, V])) #only append the Pareto Optimal solution .V: U_agent ,U: U_principal 
        #A Pareto solution is a set of multiobjective values that can be compared to other Pareto values using Pareto preference.
        #This means that a solution dominates, or is better than, another solution
        #if it is better than or equal to the other solution in all objectives and strictly better in at least one objective.
    return fitness  



###### Database associated methods  #####


#Adding final utilites and compensations to Data frame 
def creating_df(u_principal,u_agent,w_l,w_h,generation):
    num = len(u_principal)#population number
    d = {'sim':[sim_key]*num,'u_principal': u_principal, 'u_agent': u_agent,'gen':[generation]*num}
    df_sim = pd.DataFrame(data=d)
    
    global w_L_periods,w_H_periods, cols
    w_L_periods, w_H_periods = [],[]

    for i in range(contract_len):
        w_L_periods.append([x[i] for x in w_l])
        df_sim['w_l_'+str(contract_len-i)] = w_L_periods[i] 

    for i in range(contract_len):
        w_H_periods.append([x[i] for x in w_h])
        df_sim['w_h_'+str(contract_len-i)] = w_H_periods[i] 
    #Saving last simulation
    
    cols = df_sim.columns 

    return df_sim

#loading data from disc in path
def load_data(path):
    df = pd.DataFrame([], columns = [] )
    try:
        df = pd.read_csv('./data/{}.csv'.format(path))
    except:
        print("Impossible to retrieve {}.cvc".format(path))
    return df

# extracts compensations from data frame and return list by row
def get_compensations_list(data_path):
    # retrieving data fron disc
    data = load_data(data_path)
    contract_len_ = int((len(data.columns)-4)/2)
    # leaving only population of last generation 
    data = data[data['gen']==max(data['gen'])]

    # forming columns to query data
    columns = ['w_l_'+str(i+1) for i in range(contract_len_)]
    columns += ['w_h_'+str(i+1) for i in range(contract_len_)]
    # only compensations
    data = data[columns]
    # to list
    wages = data.to_numpy().tolist()
    
    return wages


# 
def simulate_dynamic():
    """This method runs a simulation with a new sim_key and current global parameters

    Returns:
        dataframe: Data of the simulation. 
        dataframe: Data of the hypervolume dataframe. 
    """
    global df_sim, sim_key

    #generating new key
    sim_key=uuid.uuid4()

    df_sim=pd.DataFrame(columns = None) 

    prng = Random()
    prng.seed(time())
    ea = inspyred.ec.emo.NSGA2(prng)
    ea.variator = [inspyred.ec.variators.blend_crossover,
                inspyred.ec.variators.gaussian_mutation]
    ea.terminator = inspyred.ec.terminators.generation_termination
    #dimensions=int(3*n_inputs / 2)
    ea.observer = [my_observer]
    ea.replacer = my_replacer

    
    # evolve is the main method of inspyred, use to Start the simulation 
    final_pop = ea.evolve(generator = custom_generator,
                        #evaluator=inspyred.ec.evaluators.parallel_evaluation_mp, #The function assigns the evaluation of each candidate (=pop_size) to its own job
                        #mp_evaluator=CustomEvaluator,
                        #mp_nprocs=4,  
                        evaluator = custom_evaluator,
                        pop_size = init_pop_size,
                        maximize = True,
                        r_inputs = n_inputs,
                        #bounder = inspyred.ec.Bounder(None, None),
                        #bounder=inspyred.ec.Bounder([0.0]*dimensions, [1.0]*dimensions),
                        max_generations = max_gen,
                        crossover_rate = crossover,
                        mutation_rate = mutation)
                        
    #final candidates(compensations) and final fitness(utilities)
    u_principal = []
    u_agent = []
    w_l = []
    w_h = []

    if display:
        final_arc = ea.archive

    #with the purpose of ploting and analysing data, this loop extracts the compensations and utilities of final generation
        for f in final_arc: 
            u_principal.append(f.fitness[0])
            u_agent.append(f.fitness[1])
            w_l.append(f.candidate[0:contract_len])
            w_h.append(f.candidate[contract_len:n_inputs])
            
    print('----------population in final arc {0}----------'.format(len(u_principal)))

    # Saving to disc   
    df_sim.to_csv('./data/{}.csv'.format(name+str(sim_key)[:7]))

    #Saving hypervolumes
    print("Saving hypervolume index ...")
    df_hyp = pd.DataFrame({'gen': generations_hv,'indicator':hypervolume_ind,'sim':[sim_key]*len(generations_hv)})
    pd.DataFrame(df_hyp).to_csv('./data/hv_{}.csv'.format(name+str(sim_key)[:7]))

    print("archive name: ", name+str(sim_key)[:7])   

    print("--------------------Ended simulation-------------------")
    

    return df_sim, df_hyp


    
##################### seeding ##########################

# setting contract length and n_inputs
def set_contract_periods(contracts):
    global contract_len, n_inputs
    contract_len = contracts 
    n_inputs = contract_len*2

# This method clear all variables and generates a new data frame 
# in which archives of new simulations will be
def reinit_variables(crra = 0.5):
    global name, df_sim, hypervolume_ind, cols
    global hv_ind_obs, generations_hv, gens_observations, gens_obs_percent
    h = crra
    # calculating cutoff generations
    gens_observations = [int(max_gen*i) for i in gens_obs_percent ]

    name = "dynamic_"+"periods"+str(contract_len)+"_"+str(max_gen)+"_h"+str(h)+"_cro_"+str(crossover)+"_mut_"+str(mutation)+"_key_"
    print('archive params :', name)
    
    # list of columns
    wls = ['w_l_'+str(contract_len-i) for i in range(contract_len)] 
    whs = ['w_h_'+str(contract_len-i) for i in range(contract_len)] 
    cols = ['sim','u_principal','u_agent','gen']  + wls + whs

    df_sim = pd.DataFrame([], columns = cols ).set_index('sim') 
    hypervolume_ind,generations_hv = [],[]
    hv_ind_obs = []


# Sets seeding mode and verify that the initial seed is adequate
def reset_seeding(seeds):
        # Setting up seed
    global init_pop_random, count_seed, init_pop_seed, init_pop_size
    # if seeding
    if not init_pop_random:
        init_pop_seed = seeds
        count_seed = 0
        print("********  Seeding mode enable ********")
    else:
        print("**** Random population mode enable ****")

    #making initial population size equal than the population seed size
    if not init_pop_random:
        init_pop_size_ = len(init_pop_seed)
        if init_pop_size == 0:
            init_pop_random = True
            print("Seeding is not possible, changing mode to random population")
        else: init_pop_size = init_pop_size_


def get_extended_dataframe(df,arch_name):
    """ Constructs utilities and efforts by periods, from a standard dataframe with wages by periods

    Args:
        df (dataframe): Dataframe of previous simulation
        name (string): Previous archive's name

    Returns:
        df_final (dataframe):  Returns a dataframe with the previous information and the new columns calculated.
        [U_low, U_high, V_low, V_high, va_periods] (list): Other arguments, low and high utility, and efforts
    """

    #Variables to manipulate 
    df_f = df[df['gen']==max(df['gen'])] #final generation

    #Sorting database by principal utility
    df_f=df_f.sort_values('u_principal')

    contract_len_ = int((len(df.columns)-3)/2)
    print("The dataframe has ",str(contract_len_), " periods" )

    #Saving by period, w_x_periods contains in i all the compensations of period i
    w_L_periods = [list(df_f['w_l_'+str(contract_len_-i)]) for i in range(contract_len_)]
    w_H_periods = [list(df_f['w_h_'+str(contract_len_-i)]) for i in range(contract_len_)]

    #Getting intermediate periods utilities and effort : U_periods, V_periods, extracting information from database

    # U defines agent utility, V defines principal utility
    U_periods,V_periods,va_periods,V_low,V_high,U_low,U_high = [],[],[],[],[],[],[]
    # length of final population
    indiv_count = len(w_H_periods[0])
    # u defines agent utility, v defines principal utility, l refers to  low output, h refers to high output
    u_l, u_h, v_l, v_h = [0]*indiv_count,[0]*indiv_count,[0]*indiv_count,[0]*indiv_count
    # copying dataframe to save a more complex version of the simulation
    df_int = df_f.copy()

    #this loop iterates the compensations and recalculates others variables(expected utilities by periods, expected utilities by periods)
    for i in range(contract_len):
        if i != 0: 
            u_l = U_low[i-1]
            u_h = U_high[i-1]
            v_l = V_low[i-1]
            v_h = V_high[i-1]
        #Recalculating efforts from compensations
        va_periods.append([validate_accion(w_H_periods[i][j],
                                        w_L_periods[i][j],
                                        v_h[j],
                                        v_l[j])[0] for j in range(indiv_count)])
        #Recalculating agent utility from compensations
        U_periods.append([principal_utility(w_H_periods[i][j],
                                        w_L_periods[i][j],
                                        va_periods[i][j],
                                        u_h[j],
                                        u_l[j]) for j in range(indiv_count)])
        #Recalculating principalutility from compensations
        V_periods.append([agent_utility(w_H_periods[i][j],
                                    w_L_periods[i][j],
                                    va_periods[i][j],
                                    v_h[j],v_l[j]) for j in range(indiv_count)])
        #Recalculating low and high utility
        U_low.append([low_principal_utility(w_L_periods[i][j]) for j in range(indiv_count)])
        U_high.append([high_principal_utility(w_H_periods[i][j]) for j in range(indiv_count)])
        V_low.append([low_agent_utility(w_L_periods[i][j],
                                    va_periods[i][j]) for j in range(indiv_count)])
        V_high.append([high_agent_utility(w_H_periods[i][j],va_periods[i][j]) for j in range(indiv_count)])
        df_int['v_'+str(contract_len_-i)] = V_periods[i]
        df_int['u_'+str(contract_len_-i)] = U_periods[i] 
        
        
    #saving intermediates utilities
    pd.DataFrame(df_int).to_csv('./data/intern_'+arch_name+'.csv')

    return df_int, [U_periods, V_periods, w_L_periods,w_H_periods,va_periods, U_low, U_high, V_low, V_high, contract_len_]