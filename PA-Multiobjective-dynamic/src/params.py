
##########################################################################################
#######                          Parmaters and variables                         #########
# This script contains the parameters of the genetic algorithm of the principal
# agent model for the asymmetric or imperfect case, 
# of the dynamic problem
##########################################################################################

##################### principal-agent variables ###################



# Contract length
contract_len = 2 

# This is the number of inputs for the evolutive algorithm, 
# they represent the number of chromosomes of each individual
# length of input = contract_len*2 (low + high)
n_inputs = contract_len*2 

# Inferior constraint, non-negativity condition
limit = 0.0   

#low outcome
YL = 2.0

#high outcome
YH = 4.0       

# discount factor
beta = 0.96    

# risk aversion coefficient
h = 0.50   

##################### genetic algorithm variables ###################

# number of generations (iterations) of the Genetic Algorithm
max_gen = 500

# crossover coefficient
crossover = 0.80

# mutation coefficient
mutation = 0.01

# initial population size
init_pop_size = 10*contract_len

#Hypervolume Indicator (HV)
hypervolume_ind = []

# generations of hypervolumes
generations_hv = []

# defines boolean variable, if the initial population is going to be seed(False) or random(True)
init_pop_random = True

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

