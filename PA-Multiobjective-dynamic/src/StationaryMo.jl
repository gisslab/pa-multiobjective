module StationaryMo

# Usings
using DelimitedFiles

############### Params  ########################################

#cara = coefficient of absolute risk averse
crra = 0.5

#beta = discount factor (same for principal and agent)
beta = 0.96

#output levels
y_l =  2# 2
y_h = 4 # 4

#h = steps or partitions of the State Variable
h = 100

#k_0 initial state
k_0 = 1

#k_f = final state
k_f = h

################ Variables ############################

#a_opt = effort in each k
a_opt =  fill(0.0, h);

# w_x = current compensation when observed output is x = low,high
w_h = fill(-0.10, h);
w_l = fill(-0.10, h);

#agent future compensation when observed output is x = low,high
V_h = fill(-1.0, h);
V_l = fill(-1.0, h);

#agent future compensation when observed output is x = low,high
U_h = fill(-1.0, h);
U_l = fill(-1.0, h);


#State Variable, b= bargaining power, takes values from 0 to 1
b = Array(range(0,step=1/(h-1),stop = 1));

#the EU_beta of the agent: if he exerts high effort and gets paid 0
V_low = 0
#the EU_beta of the agent: if he exerts low efforts and gets paid everything being produced
V_high = 0

#the EU_beta of the principal: if the agent exerts low effort and gets paid the highest possible salary
U_low = 0
#the EU_beta of the principal: if  the agent exerts high effort and gets paid the least
U_high = 0

############# BELLMAN EQUATION VARIABLES ##############3

# value of the principal expected utilities approximation given w_l, w_h, U_h, U_h, V_l,V_h,a 
U_value = fill(0.0,h) 

# value of the agent expected utilities approximation  given w_l, w_h, U_h, U_h, V_l,V_h,a 
V_value = fill(0.0,h) 

#value function, Pareto weight function
Value_function=fill(0.0,h)

# dynamic dictionaries use for memoisation of the state variables (value feasible W matrix,and efforts)
dict = Dict()
dict_com = Dict()


############################### Functions #####################################

# distribution function of high output given continuous effort
function f_continuous(y, a)
    if y == y_l
        return exp(- a)
    elseif y == y_h
        return 1- exp(-a)
    else return 0
    end
end


#future compensantion function of the agent, there's also an array below
Vf(k) = V_low+ (k-1)*(V_high-V_low)/(h-1)

#utility function of the agent, parameters consumption an effort, with cara
v_agent_cara(c,a)= -exp(cara*(a-alpha*c))

#utility function of the agent, parameters consumption an effort, with crra
v_agent_crra(c,a) = c^(1-crra)/(1-crra) - a

# setting crra
v_agent = v_agent_cara

# setting continuous distribution function
f = f_continuous#f_discrete#f_continuous



################# Constraints ###################################

#Evaluate principal utility given k_t, k_t+1, w_x,V_x
function evaluate_principal_utility(a, k_t, k_t_l, k_t_h, _w_l, _w_h, U_past)
    low = (y_l - _w_l + beta *U_past[k_t_l])*f(y_l,a)
    high = (y_h - _w_h + beta *U_past[k_t_h])*f(y_h,a)
    return low+high
end

#Evaluate agent utility given k_t, k_t+1, w_x, V_x
function evaluate_agent_utility(a, k_t, k_t_l, k_t_h, _w_l, _w_h, V_past)
    low = (v_agent(_w_l,a) + beta*V_past[k_t_l] )*f(y_l,a)
    high = (v_agent(_w_h,a) +beta *V_past[k_t_h])*f(y_h,a)
    return low+high
end

# incentive compatibility contraint for the case of 2 efforts, the continuous case returns also the effort
function incentive_compatible_continuous(a_, k_t, k_t_l, k_t_h, _w_l, _w_h, V_past)

    V = V_past
    e_a_ = (_w_h^(1 - crra))/(1 - crra) - (_w_l^(1 - crra))/(1 - crra) + beta * (V[k_t_h] - V[k_t_l])
    if e_a_ > 1 #because effort = ln(e^a), a must be positive
        a_ = log(e_a_)
        return true, a_
    else return false, a_
    end
end


#incentive compatible
incentive_compatible = incentive_compatible_continuous

############ variables and model  functions #####################

function set_parameters_crra(_crra)
    #params
    global y_l = 2
    global y_h = 4

    global crra = _crra

    # setting crra
    global v_agent = v_agent_crra

    # setting continuous distribution function
    global f = f_continuous

    global incentive_compatible = incentive_compatible_continuous

    #State Variable, b= expected discounted utility of the agent, takes values from 0 to 1
    b = Array(range(0,step=1/(h-1),stop = 1));

    # Minimum effort affects other variables, maximum effort on the continuous case does not affect other variables. 
    a_l = 0
    a_h = 2

    #the EU_beta of the agent if he exerts high effort and gets paid 0
    global V_low = v_agent(0,a_h)/(1-beta)
    #the EU_beta of the agent if he exerts low efforts and gets paid everything being produces
    # TODO: Think about that if promised compensation is high, the reservation utility can be higher
    global V_high = (v_agent(y_l,a_l)*f(y_l,a_l)+ v_agent(y_h,a_l)*f(y_h,a_l))/(1-beta)

    #the EU_beta of the principal: if the agent exerts low or high effort and gets paid the highest possible salary
    global U_low =( (y_l-y_l)*f(y_l,a_l) + (y_h-y_h)*f(y_h,a_l) )/(1-beta) #clearly 0
    #the EU_beta of the principal: if  the agent exerts high effort and gets paid the least
    global U_high = ((y_l)*f(y_l,a_h) + (y_h)*f(y_h,a_h) )/(1-beta)

    #println("len ", length(U), " h ", h)
    println(" *** setting y low = $y_l , y high = $y_h , a min = $a_l ,  a max = no upper bound , crra = $crra , crra function, $f")
    println(" *** possible utilities principal: ",U_low , " - ", U_high)
    println(" *** possible utilities agent: ", V_low, " - ",V_high)

end

# reset dynamic dictionary of compensations
function reset_compensations()
    global dict = Dict();
    global dict_com = Dict()
end

###################### MAIN ESTIMATION ##################################

##### First Approch : Naive Algorithm #####
##### Assumoming that for every state K as K_0 ######
##### you want to optimize and to know which is the next state K_1 #####

#### IDEA: Naive strategy in every state for every control, backward recursion.
#### Numerical Method: Value Function Iteration
#### Step maximization: Naive strategy, given an initial state proof with every possible a,w_h,w_l, V_l, V_h.
#### This algorithm uses memoization; for the tuples (k,k_l,k_h),saves the best w_l, w_h and effort
#### It also search in an specific direction, that is it search v-low values near v-high values
function bellman_equation_naive(iterations = 500,
    step_w = 0.005,
    Value_past = fill(0,h),
    U_past_ = fill(0,h),
    V_past_ = fill(0,h),
    window = 5,
    efficient_w = false)

    println("*************Starting Dynamic compensations Algorithm -Pareto Weight value function*************")
    #state variable b(k), k =k_0, ...,k_f
    #Assume every U(k_0) = 0 if U_past null
    crite_stop =0.0000000000000000001;
    #min distance between U_s and U_s+1
    dist_U = 1000.0
    
    # variables to update past and present iteration values 
    U_past = copy(U_past_) 
    U_current = fill(0.0,h)
    V_past = copy(V_past_)
    V_current = fill(0.0,h)
    Value_past = copy(V_past_)
    Value_current = fill(0.0,h)
    
    # other useful variables for iterations
    U_check =fill(-10.0,h)
    V_check =fill(-10.0,h)
    Value_check =fill(-10.0,h)
    # value function (Pareto Weight)
    Value_now = 0; Value_max = -1000000
    # principal 
    U_now=0; U_max =-100000;
    # agent 
    V_now=0; V_max =-100000;
    # counters and indexes
    index = 0; count =0
    # p = defines the window of search above and below kh, kl: h/p
    p = h/window
    # minimum compensation to start default = 0
    w_min = 0 # -0.4
    # related to printer
    last_iteration = false; printed = false;

    while !last_iteration
        #STOP RULE 1: for all k , U_s - U_s+1 stop criterium
        #STOP RULE 2: maximum number of iterations
        if (dist_U <crite_stop || index > iterations)
            last_iteration = true
            println("************* Final iteration ****************")
            if efficient_w
                println("********* Optimizing control grid ************")
                step_w =  step_w/5
            end
        end
        # State variable loop (b)
        for k_t in k_0:k_f

            #iterating over every possible state(k_t+1) and finding the one that gets the highest utility
            #reseting max variables
            Value_max = -100000
            U_max = -1000000
            V_max = -1000000

            # defining window search
            k_inf = (floor((k_t- h/p)) < k_0 ) ? k_0 : Int(floor((k_t- h/p)))
            k_sup = (floor((k_t+ h/p)) > k_f ) ? k_f : Int(floor((k_t+ h/p)))

            #iterating over the state vatiable in case of high output, b_h, that defines U_h(b) = U_h(k_t) = b_h = V_h(b) = V_h(k_t) 
            for k_t1_h in k_inf:k_sup

                kh_inf = (floor((k_t1_h- h/p)) < k_0 ) ? k_0 : Int(floor((k_t1_h- h/p)))

                #iterating over the state vatiable in case of low output, b_l, that defines U_l(b) = U_l(k_t) = b_l = V_l(b) = V_l(k_t) 
                for k_t1_l in kh_inf : k_t1_h
                    if  !haskey(dict,(k_t,k_t1_l,k_t1_h))
                        Value_max_w = -1000000
                        #dict[(k_t,k_t1_l,k_t1_h)] = false; continue; #parche
                        if !printed
                            println("************* Calculating optimal present compensations *************")
                            println("************* Number of steps : ********* h (V) $h **** wl ", (y_l-w_min)/step_w , "  *************")
                            printed = true
                        end

                    #     #Calculate optimum w_h,w_l, a for the tuple (k,k_l,k_h), only once, and save it on dictionary
                        for w_h_it in w_min:step_w:y_h
                           for w_l_it in w_min:step_w:y_l
                                if w_l_it > w_h_it
                                    break
                                end
                                #change IC
                                count += 1
                                # Calculating optimal effort from First order approach in incentive compatible constraint
                                # effort
                                a = -1  

                                IC, a = incentive_compatible(a,k_t,k_t1_l,k_t1_h,w_l_it,w_h_it, V_past)
                                
                                V_now = evaluate_agent_utility(a,k_t,k_t1_l,k_t1_h,w_l_it,w_h_it, V_past)

                                if (!IC) # IC must hold
                                    dict[(k_t,k_t1_l,k_t1_h)]= false
                                    continue
                                end
                                dict[(k_t,k_t1_l,k_t1_h)]= true
                                U_now =evaluate_principal_utility(a,k_t,k_t1_l,k_t1_h,w_l_it,w_h_it,U_past)
                                
                                # Value of the Pareto Weight function
                                Value_now = b[k_t] * U_now + (1-b[k_t])*V_now   
                                
                                # update if is greater than the previous 
                                if Value_now > Value_max_w #only looks for the maximum between the w's and a
                                    dict_com[(k_t,k_t1_l,k_t1_h)] = [w_l_it,w_h_it,a]
                                    Value_max_w = Value_now
                                end
                                
                            end
                        end

                    end

                    #once the a w_h w_l is calculated for the trio k,k_l,k_h
                    if dict[(k_t,k_t1_l,k_t1_h)]
                        w1 = dict_com[(k_t,k_t1_l,k_t1_h)][1]
                        w2 = dict_com[(k_t,k_t1_l,k_t1_h)][2]
                        a = dict_com[(k_t,k_t1_l,k_t1_h)][3]
                        U_now = evaluate_principal_utility(a,k_t,k_t1_l,k_t1_h,w1,w2,U_past)
                        V_now = evaluate_agent_utility(a,k_t,k_t1_l,k_t1_h,w1,w2, V_past)

                        # Value of the Pareto Weight function
                        Value_now = b[k_t] * U_now + (1-b[k_t])*V_now   

                        if Value_now > Value_max #if this combination of U(k_l, k_h) is greater than the others, update argmax
                            Value_max = Value_now
                            U_max = U_now
                            V_max = V_now
                            V_h[k_t] = k_t1_h
                            V_l[k_t] = k_t1_l
                            U_h[k_t] = k_t1_h
                            U_l[k_t] = k_t1_l
                            w_h[k_t] = w2 #w_h_it
                            w_l[k_t] = w1 #w_l_it
                            a_opt[k_t] = a
                        end
                    end
                end
            end
            # # updating values of iteration in b[k_t]
            Value_current[k_t]=Value_max
            # TODO: check  U, V current 
            U_current[k_t]=U_max
            V_current[k_t]=V_max
            if printed && index == 0
                print(" -- ($k_t) --")
                #print(" U:$(round(U_max,digits=3)) V:$(round(V_max,digits=3)) Value:$(round(Value_max,digits=3)) ")
            end
        end
        # updating values of iteration
        U_check = copy(U_past)
        U_past = copy(U_current)
        V_check = copy(V_past)
        V_past = copy(V_current)
        Value_check = copy(Value_past)
        Value_past = copy(Value_current)

        # new iteration
        index+=1

        # distance between current and last iteration
        dist_U = findmax(abs.(U_check - U_past))[1]
        sum_U = sum(abs.(U_check - U_past))

        if index%(floor(iterations/10))==0 # printer
            println("iteration : $index , max dist : $(dist_U) , sum dist : $(sum_U)")
        end
    end

    println("iterations $index , K(states) : $h , step w : $step_w , distance : $(dist_U) total order count: $count")
    println("*******Bellman Naive Algorithm has finished ******** ")

    # overwriting global variables
    global Value_function = Value_current
    global U_value = U_current
    global V_value = V_current
    return Value_current
end



############################# Saving results #################################

#base address to save files
addr_base = "./data/"

function save_results(array_, name="stationary_value_mo_$f"*"_$h.csv")
    #addr_base = pwd()
    writedlm( addr_base*name,  array_, ',')
    println("Saving values, archive  address: ",addr_base*name)
end


function save_compensations(tag = "")
    println("Saving compensations....")
    array= []
    for (k,v) in dict
        if !v
            push!(array,k[1]); push!(array,k[2]); push!(array,k[3]);
            push!(array,-1); push!(array,-1); push!(array,-1);
        else
            v = dict_com[k]
            push!(array,k[1]); push!(array,k[2]); push!(array,k[3]);
            push!(array,v[1]); push!(array,v[2]); push!(array,v[3]);
        end
    end
    addr = addr_base*"stationary_mo_compensations_$f"*"_$(h)$tag.csv"
    writedlm(addr,  array, ',')
    println("Saving dictionaries, archive address: ",addr)
    #save("compensations+$h.jld", "data", dict_com)
end


function open_results(name="stationary_mo_value_$f"*"_$h.csv")
    V_temp = readdlm(addr_base*name, ',', Float64)
    println("Loading values, archive address : ", addr_base*name)
    println("elements in list of state variable: ", length(V_temp))
    return V_temp
end


function load_compensations(tag = "")
    arch = "stationary_mo_compensations_$f"*"_$(h)$tag.csv"
    println("Loading compensations, archive name : ", arch)
    d = readdlm(addr_base*arch, ',', Float64)
    #dict_com = load("compensations+$h.jld")["data"]
    entries = Int(length(d)/6)
    for i in 1:entries
        i = 1 + (i-1)*6
        key = (Int(d[i]),Int(d[i+1]),Int(d[i+2]))
        if d[i+3]>=-0.5
            dict_com[key] = (d[i+3],d[i+4],d[i+5])
            dict[key]=true
        else dict[key]=false
        end
    end
    println("elements in dictionary : ", length(dict_com))
end





end