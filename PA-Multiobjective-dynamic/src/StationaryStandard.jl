
module StationaryStandard

# Usings
using DelimitedFiles

############### Params  ########################################
#cara = coefficient of absolute risk averse
cara = 1

#cara = coefficient of absolute risk averse
crra = 0.5

#alpha = relative cost to the agent exerting one unit of utility
alpha = 1

#beta = discount factor (same for principal and agent)
beta = 0.96

# a_x = effort levels, x = high & low
a_l = 0.1 # 0
a_h = 0.2 # 2

#output levels
y_l = 0.4 # 2
y_h = 0.8 # 4

#h = steps or partitions of the State Variable
h=100

#k_0 initial state
k_0 = 1

#k_f = final state
k_f =h

# F = Probability distribution when two efforts,
F = [1/3,2/3]

#numeric approximation
epsilon = 0.2

################ Variables ############################

#a_opt = effort in each k
a_opt =  fill(0.0, h);

# w_x = current compensation when observed output is x = low,high
w_h = fill(-0.10, h);
w_l = fill(-0.10, h);

#future compensation when observed output is x = low,high
V_h = fill(-1.0, h);
V_l = fill(-1.0, h);

#State Variable, V= expected discounted utility of the agent, takes values from V-low to V-high
#the EU_beta of the agent if he exerts high effort and gets paid 0
V_low = -exp(a_h)/(1-beta)
#the EU_beta of the agent if he exerts low efforts and gets paid everything being produces
V_high = -(1/3*exp( a_l-y_l)+ 2/3*exp(a_l-y_h))/(1-beta)

#grid point are designated uniformly over V-low and V_high
V = Array(range(V_low,step=abs(V_low-V_high)/(h-1),stop = V_high));

#### BELLMAN EQUATION VARIABLES #####

#value function
Value_function=fill(0.0,h)

###### Static case #######

# principal utility
U_static = fill(0.0,h);

#compensations static case
w_h_static = fill(0.0,h)
w_l_static = fill(0.0,h)

#effort array, possibles a when a is continuous
a_array =  Array(range(a_l,step=abs(a_l-a_h)/(20),stop = a_h)) # continuous (approximation)
a_array = [a_l, a_h] # discrete (2)

#effort static case
a_static =  fill(0.0, h);

# dynamic dictionary use for memoisation of the state variables (value feasible W matrix,and efforts)
dict = Dict()
dict_com = Dict()


############################### Functions #####################################

#f(y,l) = distribution function of outcome y given effort a
function f_discrete(y,a)
    if (((y_l==y) && (a_l==a))|| ((y_h==y) && (a_h==a)))
        return 2/3
    else return 1/3
    end
end

# distribution function of high output given continuous effort
function f_continuous(y, a)
    if y == y_l
        return exp(- a)
    elseif y == y_h
        return 1- exp(-a)
    else return 0
    end
end


#future compensantion function, there's also an array below
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

#Evaluate principal utility given k_t, k-t+1, w_x,V-x
function evaluate_principal_utility(a, k_t, k_t1_l, k_t1_h, _w_l, _w_h, U_past)
    low = (y_l - _w_l +beta *U_past[k_t1_l])*f(y_l,a)
    high = (y_h - _w_h +beta *U_past[k_t1_h])*f(y_h,a)
    return low+high
end

# individual rationality contraint
function individual_rationality(a, k_t, k_t_l, k_t_h, _w_l, _w_h)
    low = (v_agent(_w_l,a)+beta*V[k_t_l])*f(y_l,a)
    high =(v_agent(_w_h,a)+beta*V[k_t_h])*f(y_h,a)
    #if (V[k_t]+epsilon>low+high) && (V[k_t] <= low+high)
    if low + high >= V[k_t]
        return true
    else
        return false
    end
end

# incentive compatibility contraint for the case of 2 efforts
function incentive_compatible_2efforts(a_, k_t, k_t_l, k_t_h, _w_l, _w_h)
    a_other= ( a_ == a_h ) ? a_l : a_h
    u1 = (v_agent(_w_l,a_)+beta*V[k_t_l])*f(y_l,a_) + (v_agent(_w_h,a_)+beta*V[k_t_h])*f(y_h,a_)
    u2 = (v_agent(_w_l,a_other)+beta*V[k_t_l])*f(y_l,a_other) + (v_agent(_w_h,a_other)+beta*V[k_t_h])*f(y_h,a_other)
    if u1 >= u2
        return true, a_
        else return false, a_
    end
end


# incentive compatibility contraint for the case of 2 efforts, the continuous case returns also the effort
function incentive_compatible_continuous(a_, k_t, k_t_l, k_t_h, _w_l, _w_h)

    e_a_ = (_w_h^(1 - crra))/(1 - crra) - (_w_l^(1 - crra))/(1 - crra) + beta * (V[k_t_h] - V[k_t_l])
    if e_a_ > 1 #because effort = ln(e^a), a must be positive
        a_ = log(e_a_)
        return true, a_
    else return false, a_
    end
end


#incentive compatible
incentive_compatible = incentive_compatible_2efforts

############# variables and model  functions #####################

function set_parameters_crra(_crra)
    #params
    global y_l = 2
    global y_h = 4

    # TODO:  Look for a high effort that makes sense, although is not affecting state variable range
    global a_l = 0
    global a_h = 2

    global crra = _crra
    #global a_array =  Array(range(a_l,step=abs(a_l-a_h)/(effort_num),stop = a_h))
    global a_array = [a_l]

    # setting crra
    global v_agent = v_agent_crra

    # setting continuous distribution function
    global f = f_continuous

    global incentive_compatible = incentive_compatible_continuous

    #State Variable, V= expected discounted utility of the agent, takes values from V-low to V-high
    #the EU_beta of the agent if he exerts high effort and gets paid 0
    global V_low = (v_agent(0,0))/(1-beta) # if we prefer to study positive utilities, so we use effort 0 for the low V case
    #the EU_beta of the agent if he exerts low efforts and gets paid everything being produces
    global V_high = (v_agent(y_l,a_l)*f(y_l,a_l)+ v_agent(y_h,a_l)*f(y_h,a_l))/(1-beta)

    #grid point are designated uniformly over V-low and V_high
    global V = Array(range(V_low,step=abs(V_low-V_high)/(h-1),stop = V_high));

    println(" *** setting y low = $y_l , y high = $y_h , a min = $a_l ,  a max = $a_h , crra = $crra , crra function, $f")
    println(" *** reservation utilities : ", V[k_0], " - ",V[k_f])

end

function set_parameters_cara(_cara)
    #params
    global y_l = 0.4
    global y_h = 0.8

    global a_l = 0.1
    global a_h = 0.2

    global cara = _cara
    global a_array =  Array(range(a_l,step=abs(a_l-a_h)/(1),stop = a_h))

    # setting crra
    global v_agent = v_agent_cara

    # setting distribution function
    global f = f_discrete

    global incentive_compatible =incentive_compatible_2efforts

    #State Variable, V= expected discounted utility of the agent, takes values from V-low to V-high
    #the EU_beta of the agent if he exerts high effort and gets paid 0
    global V_low = -exp(a_h)/(1-beta)
    #the EU_beta of the agent if he exerts low efforts and gets paid everything being produces
    global V_high = -(1/3*exp( a_l-y_l)+ 2/3*exp(a_l-y_h))/(1-beta)

    #grid point are designated uniformly over V-low and V_high
    global V = Array(range(V_low,step=abs(V_low-V_high)/(h-1),stop = V_high));

    println(" *** setting y low = $y_l , y high = $y_h , a min = $a_l ,  a max = $a_h , cara = $cara , cara function, discrete distribution $F")
    println(" *** reservation utilities : ", V[k_0], " - ",V[k_f])
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
    U_past_ = fill(0,h),
    step_w = 0.005 ,
    window = 5, 
    efficient_w = false)

    println("*************Starting Naive Dynamic compensations Algorithm *************")
    #state variable V(k)
    #Assume every U(k_0) = 0 if U_past null
    crite_stop =0.000000000000000000001;
    #min distance between U_s and U_s+1
    dist_U = 1000.0

    # variables to update past and present iteration values 
    U_past = copy(U_past_) #Array(rand(h))#fill(1,k_f-k-0)
    U_current = fill(0.0,h)
    
    # other utils variables for iterations
    U_check =fill(-10.0,h)
    U_now=0; U_max =-100000;
    index = 0; count =0
    # p = defines the window of search above and below kh, kl: h/p
    p = h/window
    # minimum compensation to start default = 0
    w_min = 0 # -0.4
    last_iteration = false; printed = false;

    while !last_iteration
        #STOP RULE 1: for all k , U_s - U_s+1 stop criterium
        #STOP RULE 2: maximum number of iterations
        if (dist_U <crite_stop || index > iterations)
            last_iteration = true
            println("****** Final iteration *** criterium ", dist_U,"****" )
            if efficient_w
                println("********* Optimizing control grid ************")
                step_w =  step_w/5
            end
        end

        # state variable loop V
        for k_t in k_0:k_f

            #iterating for every possible state(k_t+1) and finding the one that gets the highest utility
            U_max = -1000000
            k_inf = (floor((k_t- h/p)) < k_0 ) ? k_0 : Int(floor((k_t- h/p)))
            k_sup = (floor((k_t+ h/p)) > k_f ) ? k_f : Int(floor((k_t+ h/p)))

            # loop over V_high
            for k_t1_h in k_inf:k_sup

                kh_inf = (floor((k_t1_h- h/p)) < k_0 ) ? k_0 : Int(floor((k_t1_h- h/p)))

                # loop over V_low
                for k_t1_l in kh_inf : k_t1_h
                    if  !haskey(dict,(k_t,k_t1_l,k_t1_h))
                        U_max_w = -1000000
                        #dict[(k_t,k_t1_l,k_t1_h)] = false; continue; #parche
                        if !printed
                            println("************* Calculating optimal present compensations *************")
                            println("************* Number of steps : ********* h (V) $h **** wl ", (y_l-w_min)/step_w , "  **** a ", length(a_array),"************")
                            printed = true
                        end

                    #   # Finds optimum w_h ,w_l, a for the tuple (k,k_l,k_h), only once, then, save it on dictionary
                        for w_h_it in w_min:step_w:y_h
                           for w_l_it in w_min:step_w:y_l
                                if w_l_it > w_h_it
                                    break
                                end
                                #change IC

                                for a in a_array
                                    count += 1
                                    IC, a = incentive_compatible(a,k_t,k_t1_l,k_t1_h,w_l_it,w_h_it)
                                    #a= (IC) ? a_h : a_l
                                    IR = individual_rationality(a,k_t,k_t1_l,k_t1_h,w_l_it,w_h_it)
                                    if (!IR) || (!IC)
                                        dict[(k_t,k_t1_l,k_t1_h)]= false
                                        continue
                                    end
                                    dict[(k_t,k_t1_l,k_t1_h)]= true
                                    U_now =evaluate_principal_utility(a,k_t,k_t1_l,k_t1_h,w_l_it,w_h_it,U_past)
                                    if U_now > U_max_w #only looks for the maximum between the w's and a
                                        dict_com[(k_t,k_t1_l,k_t1_h)] = [w_l_it,w_h_it,a]
                                        U_max_w = U_now
                                    end
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
                        if U_now > U_max #if this combination of U(k_l, k_h) is greater than the others, update argmax
                            U_max = U_now
                            V_h[k_t] = k_t1_h
                            V_l[k_t] = k_t1_l
                            w_h[k_t] = w2 #w_h_it
                            w_l[k_t] = w1 #w_l_it
                            a_opt[k_t] = a
                        end
                    end
                end
            end
            U_current[k_t]=U_max
            if printed && index == 0
                print(" -- ($k_t) --")
            end
        end
        # updating
        U_check = copy(U_past)
        U_past = copy(U_current)
        # new iteration 
        index+=1
        # distance btern current and last iteration
        dist_U = findmax(abs.(U_check - U_past))[1]
        sum_U = sum(abs.(U_check - U_past))
        # printer
        if index%(floor(iterations/20))==0
            println("iteration : $index , max dist : $(dist_U) , sum dist : $(sum_U)")
        end
    end

    println("iterations $index , K(states) : $h , step w : $step_w , distance : $(dist_U) total order count: $count")
    println("*******Bellman Naive Algorithm has finished ******** ")
    global Value_function = U_current
    return U_current
end


# Static case for comparison, (no future compensations)
function static_value(step_w)
    println("*************Starting Static Optimization Algorithm *************")
    for k in k_0:k_f
        U_max_w = 0.0
        #check for every w_h,w_l,a, which one solve the static problem
        for w_h_it in 0:step_w:y_h
           for w_l_it in 0:step_w:y_l
                if w_l_it>w_h_it
                    break
                end
                for a in a_array
                    #Incentive_Compatible
                    U_a = v_agent(w_l_it,a)*f(y_l,a) + v_agent(w_h_it,a)*f(y_h,a) #agent's utility
                    a_oth = (a==a_h) ? a_l : a_h
                    U_a_oth = v_agent(w_l_it,a_oth)*f(y_l,a_oth) + v_agent(w_h_it,a_oth)*f(y_h,a_oth)
                    if U_a_oth > U_a #check IC constraint
                        continue
                    end

                    #Individual_Rationality
                    if !(U_a >= (1-beta)V[k])
                        continue
                    end
                    #Principal's utility, value function
                    U_now = (y_l - w_l_it)*f(y_l,a)+ (y_h - w_h_it)*f(y_h,a)
                    if (k==60)
                        #println("k $k w_H $w_h_it w_L $w_l_it U_a $U_a V[k] $((1-beta)V[k]) U_now $U_now U_max $U_max_w")
                    end
                    if U_now > U_max_w #only looks for the maximum between the w's and a
                        w_h_static[k] = w_h_it
                        w_l_static[k] = w_l_it
                        a_static[k] = a
                        U_max_w = U_now
                    end
                end
            end
        end
        print(" -- $k --")
        U_static[k] = U_max_w
    end
    return U_static
end

############################# Saving results #################################

#base address to save files
addr_base = "./data/"

function save_results(array_, name="stationary_value_$f"*"_$h.csv")
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
    addr = addr_base*"stationary_compensations_$f"*"_$(h)$tag.csv"
    writedlm(addr,  array, ',')
    println("Saving dictionaries, archive address: ",addr)
    #save("compensations+$h.jld", "data", dict_com)
end


function open_results(name="stationary_value_$f"*"_$h.csv")
    V_temp = readdlm(addr_base*name, ',', Float64)
    println("Loading values, archive address : ", addr_base*name)
    println("elements in list of state variable: ", length(V_temp))
    return V_temp
end


function load_compensations(tag = "")
    arch = "stationary_compensations_$f"*"_$(h)$tag.csv"
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
