
##########################################################################################
#######                          Parmaters and variables                         #########
# This script has the parameters of the genetic algorithm of the principal
# agent model for the symmetric or perfect case, and the 
# assymetric or imperfect case.
##########################################################################################

##################### principal-agent variables ###################

# inferior limit, non-negativity condition
limit = 0.0  

# low output
YL = 2.0     

# high output
YH = 4.0       

# absolute risk averse coefficient (cara)
h = 0.50

##################### genetic algorithm variables ###################

# This is the number of inputs for the evolutive algorithm, 
# they represent the number of chromosomes of each individual
# static case, is always two
n_inputs = 2

# number of generations (iterations) of the Genetic Algorithm
max_gen = 1000

# size of the initial population
init_pop_size = 50

# crossover coefficient
crossover = 1.0

# mutation coefficient
mutation = 0.2

#Hypervolume Indicator (HV)
hypervolume_ind = []

# generations of hypervolumes
generations_hv = []

# hv of observations
hv_ind_obs = []

# defines boolean variable, if the initial population is going to be seed(False) or random(True)
init_pop_random = False

#defines initial population seed to use if init_pop_random is False
init_pop_seed = []

#defines a counter for retrieving seeds
count_seed = 0

#selected list of generations for observer in percent, max_gen*percent = generation
gens_obs_percent = [0,0.03,0.08,0.15,0.25,0.45,0.65,0.85,1]

#list of actual generations to observe, e.g. [0,1000,2000]
gens_observations = []

# unique key of current simulation
sim_key = 0