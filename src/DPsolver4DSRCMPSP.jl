##### 0.6 to 1.0 changes
#find turned to findall
#sum(MPTD,2) turned sum(MPTD,dims=2)
#TS[nn,:] become for nnn = 1: size(TS[nn,:],1) TS[nn,nnn]=-1 end
#using time causes problems use ttime instead
##indmax and inmin become argmax and argmin
##ind2sub is not working argmax is used
#####
#Pkg.add("Combinatorics"), #Pkg.add("StatsBase")
using Combinatorics #for conbinations chosing alternative actions.
using StatsBase # this is for using weith in mutation
#using Base.Test #not working but I forgot that why I am using it
#Pkg.add("PyPlot")
#using PyPlot #Currently I dont have any plot
#Pkg.add("JLD2") #Pkg.add("FileIO")
using JLD2, FileIO
#this is for saving data and not working anymore
import NaNMath #for what?
#Pkg.add("NaNMath") #Pkg.add("Distributions") #Pkg.add("SoftGlobalScope")
using Distributions #for exponential distribution
#using SoftGlobalScope #for solving the global scope problem
#Pkg.add("ElasticArrays")
#using ElasticArrays
    ##### 5 project sample
    #MPTD =Int8[5 1;4 2;3 3;2 4;1 2] # project task durations
    #MPRU =Int8[2 1;2 1;2 1;2 1;2 1] #project resource usage
    #PDD= Int8[4,5,6,7,8] # project due dates
    #reward=Int8[15 3;12 15;9 9;6 12;15 3]
    #Tardiness=Int8[3,4,5,6,7]
    #### 4 project sample 2 tasks
    #MPTD =Int8[5 1;4 2;3 3;2 4]#;1 5] # project task durations
    #MPRU =Int8[2 1;2 1;2 1;2 1] #project resource usage
    #PDD= Int8[4,5,6,7] # project due dates
    #Tardiness=Int8[3,4,5,6]
    #reward=Int8[0 18;0 27;0 18;0 18]#;15 3]
    ### 2 project 3 tasks sample
    #MPTD =Int8[1 2 5;4 3 4]#;7 8 9] # project task durations
    #MPRU =Int8[1 2 1;1 2 1]#;1 2 1] #project resource usage
    #PDD= Int8[10,15]#,25] # project due dates
    #Tardiness=Int8[8,5]#,25]
    #reward=Int8[0 0 12;0 0 6]#;0 0 27]
    ############Common values
    #### 3 project sample 2 tasks
    #MPTD =Int8[5 2;1 3;2 7] # project task durations
    #MPRU =Int8[1 1;2 1;3 2] #project resource usage
    #PDD= Int8[10,8,10] # project due dates
    #Tardiness=Int8[5,3,19]
    #reward=Int8[0 8;0 5;0 20]

    #### 2 project sample 2 tasks
    MPTD =Int8[2 2;3 1] # project task durations
    MPRU =Int8[2 2;1 3] #project resource usage
    PDD= Int8[8,5] # project due dates
    Tardiness=Int8[1,9]
    reward=Int8[0 3;0 10]
    #### 1 project sample 2 tasks
    #MPTD =Int8[3 5] # project task durations
    #MPRU =Int8[2 2] #project resource usage
    #PDD= Int8[8] # project due dates
    #Tardiness=Int8[1]
    #reward=Int8[0 3]
    #### 2 project sample 2 tasks
    #MPTD =Int8[5 4;5 4] # project task durations
    #MPRU =Int8[2 2;1 3] #project resource usage
    #PDD= Int8[10,10] # project due dates
    #Tardiness=Int8[1,9]
    #reward=Int8[0 3;0 10]



#'#### Add distributin to task durations

    ####IMPORTANT NOTE If you change the project network change the reward calculations in GAScore and Bellman
    Res1 = 3 #Amount of maximum resource type 1
    max_project = 5#to solve this project number requires bigger ram than 16 gb or some settings(I am not sure)
    EFoptions = 1 #Early Finish Options. early finish, normal finish, late finish
    ELFinishes=EFoptions-1 #I used this on for loups
    for P = 1:size(MPTD,1) #this adds the late completiosn
        for T =1:size(MPTD,2)
            MPTD[P,T]+=ELFinishes/2
        end
    end
        ##############################
        Value = ZerosFloat64()
        HoldValue = ZerosFloat64()  #prevents from overwriting durung the state value calculation
        ArrivalProbabilty=0.60#New project arrival probability
        ArrivalProb = [ArrivalProbabilty,ArrivalProbabilty,ArrivalProbabilty,ArrivalProbabilty,ArrivalProbabilty]#arrival prob for diffirent projects
        #policy = ZerosINT32() #int 32 only supports 5 project 2 task or 3 project and 3 task
        #policy = load("policy.jld2")["policy"]
        discount=1 #discount factor
        MaxW=999999#maximum value increases by iteration
        NEWState = AnyArray()
        NEWProb = AnyArray()
        NEWTaskProb = AnyArray()
        NEWDueDates = AnyArray()
        NEWTaskDates = AnyArray()
        DDSC = ZerosINT() #creates the array for control rule slacks
        #Controlrule() #it is not needed for deterministic problem
        @time EachProbscases()

        #cases() #this is for running for one iteration
        #save("policy.jld2", "policy", policy) # saving the policy of one iteration
        #println("done") #done of one iteration (GA and RBA)
        @time main()
sum(1+1)


    function Controlrule()
        for P = 1:size(MPTD,1)# P project no
            for T = 1:sum(MPTD,dims=2)[P]+1 #T task code
                if  T == 1 #if task is not started
                    DDSC[P,T] = 0 # there is no early start gab
                elseif T-1 <= MPTD[P]-1#if task one is processing
                    DDSC[P,T] = 0
                elseif T-1 == MPTD[P]#if task one is processing
                    DDSC[P,T] = 2
                elseif T-1 <= MPTD[P,1]+MPTD[P,2]-1#if task 2 is active
                    DDSC[P,T] = 2
                elseif T-1 == MPTD[P,1]+MPTD[P,2]#if task 2 is active
                    DDSC[P,T] = 4
                elseif T-1 <= MPTD[P,1]+MPTD[P,2]+MPTD[P,3]-1#if task 3 is active
                    DDSC[P,T] = 4
                elseif T-1 == MPTD[P,1]+MPTD[P,2]+MPTD[P,3]#if task 3 is active
                    DDSC[P,T] = 6
                elseif T-1 <= MPTD[P,1]+MPTD[P,2]+MPTD[P,3]+MPTD[P,4]#if task 4 is active
                    DDSC[P,T] = 4
                elseif T-1 <= MPTD[P,1]+MPTD[P,2]+MPTD[P,3]+MPTD[P,4]+MPTD[P,5] #if task 5 is active
                    DDSC[P,T] = 5
                end
            end
        end
    end #fixes the due dates in case of stochastic task durations (dont use if task are deterministic)
    function ZerosINT32()
        Returnvalue = 0
        if size(PDD,1) ==5
           Returnvalue = zeros(Int32,sum(MPTD[1,:])+1,PDD[1]+1,sum(MPTD[2,:])+1,
           PDD[2]+1,sum(MPTD[3,:])+1,PDD[3]+1,sum(MPTD[4,:])+1,PDD[4]+1,sum(MPTD[5,:])+1,PDD[5]+1)
        elseif size(PDD,1) ==4
            Returnvalue = zeros(Int32,sum(MPTD[1,:])+1,PDD[1]+1,sum(MPTD[2,:])+1,
            PDD[2]+1,sum(MPTD[3,:])+1,PDD[3]+1,sum(MPTD[4,:])+1,PDD[4]+1,1,1)
        elseif  size(PDD,1) ==3
            Returnvalue = zeros(Int32,sum(MPTD[1,:])+1,PDD[1]+1,sum(MPTD[2,:])+1,
            PDD[2]+1,sum(MPTD[3,:])+1,PDD[3]+1,1,1,1,1)
        elseif  size(PDD,1) ==2
            Returnvalue = zeros(Int32,sum(MPTD[1,:])+1,PDD[1]+1,sum(MPTD[2,:])+1,PDD[2]+1,1,1,1,1,1,1)
        elseif  size(PDD,1) ==1
            Returnvalue = zeros(Int32,sum(MPTD[1,:])+1,PDD[1]+1,1,1,1,1,1,1,1,1)
        end
        return Returnvalue
    end #creates a full state matrix(each task each project) for policies
    #=function ZerosINT8()
        Returnvalue = 0
        if size(PDD,1) ==5
           Returnvalue = zeros(Int32,sum(MPTD[1,:])+1,sum(MPTD[2,:])+1,
          sum(MPTD[3,:])+1,sum(MPTD[4,:])+1,sum(MPTD[5,:])+1)
        elseif size(PDD,1) ==4
            Returnvalue = zeros(Int32,sum(MPTD[1,:])+1,sum(MPTD[2,:])+1,
            sum(MPTD[3,:])+1,sum(MPTD[4,:])+1,1)
        elseif  size(PDD,1) ==3
            Returnvalue = zeros(Int32,sum(MPTD[1,:])+1,sum(MPTD[2,:])+1,
            sum(MPTD[3,:])+1,1,1)
        elseif  size(PDD,1) ==2
            Returnvalue =  zeros(Int32,sum(MPTD[1,:])+1,sum(MPTD[2,:])+1,1,1,1)
        elseif  size(PDD,1) ==1
            Returnvalue = zeros(Int32,sum(MPTD[1,:])+1,1,1,1,1)
        end
        return Returnvalue
    end =##creates policies matrix use int32
    function ZerosINT()
        Returnvalue = 0
        if size(PDD,1) ==5
           Returnvalue = zeros(Int32,5,sum(MPTD[1,:])+1+sum(MPTD[2,:])+1+
          sum(MPTD[3,:])+1+sum(MPTD[4,:])+1+sum(MPTD[5,:])+1)
        elseif size(PDD,1) ==4
            Returnvalue = zeros(Int32,4,sum(MPTD[1,:])+1+sum(MPTD[2,:])+1+
            sum(MPTD[3,:])+1+sum(MPTD[4,:])+1)
        elseif  size(PDD,1) ==3
            Returnvalue = zeros(Int32,3,sum(MPTD[1,:])+1+sum(MPTD[2,:])+1+
            sum(MPTD[3,:])+1)
        elseif  size(PDD,1) ==2
            Returnvalue =  zeros(Int32,2,sum(MPTD[1,:])+1+sum(MPTD[2,:])+1)
        elseif  size(PDD,1) ==1
            Returnvalue = zeros(Int32,1,sum(MPTD[1,:])+1)
        end
        return Returnvalue
    end #used for control rule(fixing due date on stochastic tasks) for each state
    function AnyArray()
        Returnvalue = 0
        if size(PDD,1) ==5
           Returnvalue = Array{Any}(undef,sum(MPTD[1,:])+1+size(MPTD,2),
           sum(MPTD[2,:])+1+size(MPTD,2),sum(MPTD[3,:])+1+size(MPTD,2),
           sum(MPTD[4,:])+1+size(MPTD,2),sum(MPTD[5,:])+1+size(MPTD,2))
        elseif size(PDD,1) ==4
            Returnvalue = Array{Any}(undef,sum(MPTD[1,:])+1+size(MPTD,2),
            sum(MPTD[2,:])+1+size(MPTD,2),sum(MPTD[3,:])+1+size(MPTD,2),
            sum(MPTD[4,:])+1+size(MPTD,2),1)
        elseif  size(PDD,1) ==3
            Returnvalue = Array{Any}(undef,sum(MPTD[1,:])+1+size(MPTD,2),
            sum(MPTD[2,:])+1+size(MPTD,2),sum(MPTD[3,:])+1+size(MPTD,2),1,1)
        elseif  size(PDD,1) ==2
            Returnvalue =  Array{Any}(undef,sum(MPTD[1,:])+1+size(MPTD,2),
            sum(MPTD[2,:])+1+size(MPTD,2),1,1,1)
        elseif  size(PDD,1) ==1
            Returnvalue = Array{Any}(undef,sum(MPTD[1,:])+1+size(MPTD,2),1,1,1,1)
        end
        return Returnvalue
    end #creates a full state matrix(each task each project) for any info
    #=function ZerosFloat16()
        Returnvalue = 0
        if size(PDD,1) ==5
           Returnvalue = zeros(Float16,sum(MPTD[1,:])+1,sum(MPTD[2,:])+1,
          sum(MPTD[3,:])+1,sum(MPTD[4,:])+1,sum(MPTD[5,:])+1)
        elseif size(PDD,1) ==4
            Returnvalue = zeros(Float16,sum(MPTD[1,:])+1,sum(MPTD[2,:])+1,
            sum(MPTD[3,:])+1,sum(MPTD[4,:])+1,1)
        elseif  size(PDD,1) ==3
            Returnvalue = zeros(Float16,sum(MPTD[1,:])+1,sum(MPTD[2,:])+1,
            sum(MPTD[3,:])+1,1,1)
        elseif  size(PDD,1) ==2
            Returnvalue =  zeros(Float16,sum(MPTD[1,:])+1,sum(MPTD[2,:])+1,1,1,1)
        elseif  size(PDD,1) ==1
            Returnvalue = zeros(Float16,sum(MPTD[1,:])+1,1,1,1,1)
        end
        return Returnvalue
    end=# #creates value matrix of state space (creates error use Float64)
    function ZerosFloat64()
        Returnvalue = 0
        if size(PDD,1) ==5
           Returnvalue = zeros(Float64,sum(MPTD[1,:])+1,PDD[1]+1,sum(MPTD[2,:])+1,
           PDD[2]+1,sum(MPTD[3,:])+1,PDD[3]+1,sum(MPTD[4,:])+1,PDD[4]+1,sum(MPTD[5,:])+1,PDD[5]+1)
        elseif size(PDD,1) ==4
            Returnvalue = zeros(Float64,sum(MPTD[1,:])+1,PDD[1]+1,
            sum(MPTD[2,:])+1,PDD[2]+1,sum(MPTD[3,:])+1,PDD[3]+1,sum(MPTD[4,:])+1,PDD[4]+1)
        elseif  size(PDD,1) ==3
            Returnvalue = zeros(Float64,sum(MPTD[1,:])+1,PDD[1]+1,
            sum(MPTD[2,:])+1,PDD[2]+1,sum(MPTD[3,:])+1,PDD[3]+1,1,1)
        elseif  size(PDD,1) ==2
            Returnvalue = zeros(Float64,sum(MPTD[1,:])+1,PDD[1]+1,sum(MPTD[2,:])+1,PDD[2]+1,1,1,1,1)
        elseif  size(PDD,1) ==1
        Returnvalue = zeros(Float64,sum(MPTD[1,:])+1,PDD[1]+1,1,1,1,1,1,1)
        end
        return Returnvalue
    end #creates value matrix of state space
    function disp(state,P,T)
        #state,P,T = 2,1,1
        #here we have uniform distribution
        EFprob = 0.000
        if MPTD[P,T]!=2 #if state late finish is not 2, #normal finish is 1, early finish is 0
            if state > 0 && EFoptions >= state
            EFprob = 1/state
            end
        else
            if state == 2
                EFprob = 1/3
            elseif state == 1
                EFprob = 2/3
            end
        end
        return EFprob
    end # creates uniform distribution of early completion (1/3,1/2,1)
    function EachProbscases()
        P2,P3,P4,P5=1,1,1,1
        for P1 = 1:sum(MPTD,dims=2)[1]+1+size(MPTD,2) #last +size(MPTD,2) for active but not started task
            if size(PDD,1) > 1 #If more than 2 project exist
                for P2 = 1:sum(MPTD,dims=2)[2]+1+size(MPTD,2) #last +size(MPTD,2) for active but not started task
                    if size(PDD,1) > 2# If more than 3 project exist
                        for P3 = 1:sum(MPTD,dims=2)[3]+1+size(MPTD,2) #last +size(MPTD,2) for active but not started task
                            if size(PDD,1) > 3# If more than 4 project exist
                                for P4 = 1:sum(MPTD,dims=2)[4]+1+size(MPTD,2) #last +size(MPTD,2) for active but not started task
                                    if size(PDD,1) > 4# If more than 4 project exist
                                        for P5 = 1:sum(MPTD,dims=2)[5]+1+size(MPTD,2) #last +size(MPTD,2) for active but not started task
                                            NextState(P1,P2,P3,P4,P5)
                                        end
                                    else
                                        NextState(P1,P2,P3,P4,P5)
                                    end# if only 4 project exist
                                end
                            else
                                NextState(P1,P2,P3,P4,P5)
                            end#if only 3 project exist
                        end
                    else
                        NextState(P1,P2,P3,P4,P5)
                    end#if only 2 project exist
                end
            else
                NextState(P1,P2,P3,P4,P5)
            end #if only one project exist
        end
    end #finds future states #suitable for any number of project with any number of task # not ready for any task order
    function NextState(P1,P2,P3,P4,P5) #SS is not next state but action and State together
        #P1,P2,P3,P4,P5 = 7,1,1,1,1
        SS = stateConventor2(P1,P2,P3,P4,P5)
        DD = zeros(Int8,max_project)#Due dates changes #0 = no change, 1= iterate due date, 2= project finished, 3=project arrived#
        result=[] #next state of early arrivals
        prob=[]#arrival and task durations probability
        Taskprob=[]#task durations probability
        Taskdates=[]#task for reward calculation
        TaskCompletionP=[]#stochastic task completion project holder
        TaskCompletionT=[]#stochastic task completion task holder
        times=[] #remained due dates for  each option of arrivals
        a = zeros(Int8, size(SS,1),size(SS,2))#Action here is do nothing, this is only used to iterate the state to future, real action is already added.
        ##Selecting of possible early finishes
        SomeProbabilty = 1.00
        State_V_plus_one = copy(SS)
        for  projectno = 1:size(SS,1) #for each project
            for taskno = 1:size(SS,2)#for each task
                if SS[projectno,taskno] <= EFoptions &&  SS[projectno,taskno] > 1#if project is in last 3 round
                push!(TaskCompletionP,projectno) #save as projectno of earlyfinish
                push!(TaskCompletionT,taskno) #save as projectno of earlyfinish
                end
                ###state iterating without change
                if SS[projectno,taskno] > 0
                    State_V_plus_one[projectno,taskno]=SS[projectno,taskno]-1
                    DD[projectno]=1 #iterate due date
                    if State_V_plus_one[projectno,taskno]==0#if project is finished
                        DD[projectno]=2#due date become -1
                    end
                end
                if SS[projectno,taskno] == -1
                    DD[projectno]=1 #iterate due date
                end
            end
        end
        ####NO EARLY START OPTION
        #push!(result1,conventor(State_V_plus_one)) #Saving case which no new project arrives.
        for P in TaskCompletionP
            T = TaskCompletionT[findall(x->x==P , TaskCompletionP)[1]]#gives the referance of early completed task
            #=global=# #SomeProbabilty *=(1-EarlyC[P,T,(SS[P,T]-1)]) #not Early completion prob
            #=global=# SomeProbabilty *=(1-disp(SS[P,T],P,T))#not Early completion prob  #with function
        end
        push!(Taskprob,SomeProbabilty) #SaveTaskprob
        push!(Taskdates,DD)
        #push!(TaskCompProb,SomeProbabilty) #Saving the prossbilty of non early finish
        #push!(times,t) #Remained due date matrix for no early finish
        prob,result,times = ProbabityOptions2(prob,result,times,State_V_plus_one,SomeProbabilty,DD)
        #####
        ####EARLY START OPTION
        for N in collect(combinations(TaskCompletionP)) #finding all task completion #Not suitable for every task matrix
            SomeProbabilty=1
            TS = copy(State_V_plus_one)
            RD = copy(DD)#remaining due date, this is solve a misterious overwriting problem.
            for P in N #Creates remained due dates and rewards for early endings
                T = TaskCompletionT[findall(x->x==P , TaskCompletionP)[1]] #gives the referance of early completed task
                TS[P,T]=0#finishing the task early
                if T == size(SS[P,:],1)##if this task is the last task of type nn project
                    RD[P] = 2 #makes the early completed projects duration due
                end
            end
            # this creates the probabilities of early completion
            for P in TaskCompletionP #it consider all early completable project not only early completed ones like N, So it calculates both early finishing and not finishing together
                T = TaskCompletionT[findall(x->x==P , TaskCompletionP)[1]]#gives the referance of early completed task
                if P in N
                    #SomeProbabilty = SomeProbabilty*EarlyC[P,T,(SS[P,T]-1)] #Early completion prob
                    SomeProbabilty = SomeProbabilty*disp(SS[P,T],P,T) #Early completion prob #with function
                else
                    #SomeProbabilty = SomeProbabilty*(1-EarlyC[P,T,(SS[P,T]-1)]) #not Early completion prob
                    SomeProbabilty = SomeProbabilty*(1-disp(SS[P,T],P,T)) #not Early completion prob  #with function
                end
            end
            push!(Taskprob,SomeProbabilty) #possibility distribution for each option of arrivals
            push!(Taskdates,RD)
            #push!(result1,conventor(TS)) #NEXT STATE early completion
            #push!(times,tm) #remained due dates for  early completion
            #=global=# prob,result,times = ProbabityOptions2(prob,result,times,TS,SomeProbabilty,RD)
        end
        NEWState[P1,P2,P3,P4,P5] = result
        NEWProb[P1,P2,P3,P4,P5] = prob
        NEWTaskProb[P1,P2,P3,P4,P5] = Taskprob
        NEWTaskDates[P1,P2,P3,P4,P5] = Taskdates
        NEWDueDates[P1,P2,P3,P4,P5] = times
    end #finds the next states(and their probs with and without arrivals) of given post decision state,
    #=function Probscases()
        c,d,e,f,g,h,k,l=1,1,1,1,1,1,1,1
            for a = 1 : PDD[1]+1
                for b = 1:sum(MPTD,dims=2)[1]+1
                    if  a <= PDD[1]-(b-1)+ELFinishes*size(MPTD,2) || a > PDD[1]
                        if size(PDD,1) > 1 #If more than 2 project exist
                            for c = 1 : PDD[2]+1
                                for d = 1:sum(MPTD,dims=2)[2]+1
                                    if  c <= PDD[2]-(d-1)+ELFinishes*size(MPTD,2) || c > PDD[2]
                                        if size(PDD,1) > 2# If more than 3 project exist
                                            for e = 1 : PDD[3]+1
                                                for f = 1:sum(MPTD,dims=2)[3]+1
                                                    if e <= PDD[3]-(f-1)+ELFinishes*size(MPTD,2) || e > PDD[3]
                                                        if size(PDD,1) > 3# If more than 4 project exist
                                                            for g = 1 : PDD[4]+1
                                                                for h = 1:sum(MPTD,dims=2)[4]+1
                                                                    if  g <= PDD[4]-(h-1)+ELFinishes*size(MPTD,2) || g > PDD[4]
                                                                        if size(PDD,1) > 4# If more than 4 project exist
                                                                            for k = PDD[5]+1
                                                                                for l = 1:sum(MPTD,dims=2)[5]+1
                                                                                    if k <= PDD[5]-(l-1)+ELFinishes*size(MPTD,2) || k > PDD[5]
                                                                                        ProbsFounder(a,b,c,d,e,f,g,h,k,l)
                                                                                    end
                                                                                end
                                                                            end
                                                                        else
                                                                            ProbsFounder(a,b,c,d,e,f,g,h,k,l)
                                                                        end# if only 4 project exist
                                                                    end
                                                                end
                                                            end
                                                        else
                                                            ProbsFounder(a,b,c,d,e,f,g,h,k,l)
                                                        end#if only 3 project exist
                                                    end
                                                end
                                            end
                                        else
                                            #println(a,b,c,d,e,f,g,h,k,l)
                                            ProbsFounder(a,b,c,d,e,f,g,h,k,l)
                                        end#if only 2 project exist
                                    end
                                end
                            end
                        else
                            ProbsFounder(a,b,c,d,e,f,g,h,k,l)
                        end #if only one project exist
                    end
                end
            end
    end =##not active #suitable for any number of project with any number of task # not ready for any task order
    function stateConventor(p1,p2,p3,p4,p5)
        #i,j,k,l = 8,14,99,99# test purpose
        #MPTD =Int8[1 2 3;4 5 6]# test purpose
        Task_states = zeros(Int8,size(PDD,1),size(MPTD,2))
        pstatecode = [p1,p2,p3,p4,p5]
        for projectno = 1:size(PDD,1)
            if  pstatecode[projectno] == 1
                for task = 1:size(MPTD,2)
                    Task_states[projectno,task] = -1
                end
            elseif pstatecode[projectno]-1 <= MPTD[projectno]
                Task_states[projectno,1] = MPTD[projectno]-(pstatecode[projectno]-1)
                if size(MPTD,2) > 1
                    for task = 2:size(MPTD,2)
                        Task_states[projectno,task] = -1
                    end
                end
            elseif pstatecode[projectno]-1 <= MPTD[projectno,1]+MPTD[projectno,2]
                Task_states[projectno,1] = 0
                Task_states[projectno,2]= MPTD[projectno,1]+MPTD[projectno,2]-(pstatecode[projectno]-1)
                if size(MPTD,2) > 2
                    for task = 3:size(MPTD,2)
                        Task_states[projectno,task] = -1
                    end
                end
            elseif pstatecode[projectno]-1 <= MPTD[projectno,1]+MPTD[projectno,2]+MPTD[projectno,3]
                Task_states[projectno,1] = 0
                Task_states[projectno,2] = 0
                Task_states[projectno,3]= MPTD[projectno,1]+MPTD[projectno,2]+MPTD[projectno,3]-(pstatecode[projectno]-1)
                if size(MPTD,2) > 3
                    for task = 4:size(MPTD,2)
                        Task_states[projectno,task] = -1
                    end
                end
            elseif pstatecode[projectno]-1 <= MPTD[projectno,1]+MPTD[projectno,2]+MPTD[projectno,3]+MPTD[projectno,4]
                Task_states[projectno,1] = 0
                Task_states[projectno,2] = 0
                Task_states[projectno,3] = 0
                Task_states[projectno,4]= MPTD[projectno,1]+MPTD[projectno,2]+MPTD[projectno,3]+MPTD[projectno,4]-(pstatecode[projectno]-1)
                if size(MPTD,2) > 4
                    for task = 5:size(MPTD,2)
                        Task_states[projectno,task] = -1
                    end
                end
            elseif pstatecode[projectno]-1 <= MPTD[projectno,1]+MPTD[projectno,2]+MPTD[projectno,3]+MPTD[projectno,4]+MPTD[projectno,5]
                Task_states[projectno,1] = 0
                Task_states[projectno,2] = 0
                Task_states[projectno,3] = 0
                Task_states[projectno,4] = 0
                Task_states[projectno,5]= MPTD[projectno,1]+MPTD[projectno,2]+MPTD[projectno,3]+MPTD[projectno,4]+MPTD[projectno,5]-(pstatecode[projectno]-1)
            end
        end
        return Task_states
    end #change format of pre-desision state space (e.g., 7 => (0,5)) #task order needed
    function stateConventor2(p1,p2,p3,p4,p5)
        #p1,p2,p3,p4,p5 = 13,1,1,1,1# test purpose
        #MPTD =Int8[5 4 3;5 4 3] # test purpose
        Task_states = zeros(Int8,size(PDD,1),size(MPTD,2))
        pstatecode = [p1,p2,p3,p4,p5]
        for projectno = 1:size(PDD,1)
            if  pstatecode[projectno] == 1
                for task = 1:size(MPTD,2)
                    Task_states[projectno,task] = -1
                end
            elseif pstatecode[projectno]-1 <= MPTD[projectno]+1
                Task_states[projectno,1] = MPTD[projectno]-(pstatecode[projectno]-2)
                if size(MPTD,2) > 1
                    for task = 2:size(MPTD,2)
                        Task_states[projectno,task] = -1
                    end
                end
            elseif pstatecode[projectno]-1 <= MPTD[projectno,1]+MPTD[projectno,2]+2
                Task_states[projectno,1] = 0
                Task_states[projectno,2]= MPTD[projectno,1]+MPTD[projectno,2]-(pstatecode[projectno]-3)
                if size(MPTD,2) > 2
                    for task = 3:size(MPTD,2)
                        Task_states[projectno,task] = -1
                    end
                end
            elseif pstatecode[projectno]-1 <= MPTD[projectno,1]+MPTD[projectno,2]+MPTD[projectno,3]+3
                Task_states[projectno,1] = 0
                Task_states[projectno,2] = 0
                Task_states[projectno,3]= MPTD[projectno,1]+MPTD[projectno,2]+MPTD[projectno,3]-(pstatecode[projectno]-4)
                if size(MPTD,2) > 3
                    for task = 4:size(MPTD,2)
                        Task_states[projectno,task] = -1
                    end
                end
            elseif pstatecode[projectno]-1 <= MPTD[projectno,1]+MPTD[projectno,2]+MPTD[projectno,3]+MPTD[projectno,4]+4
                Task_states[projectno,1] = 0
                Task_states[projectno,2] = 0
                Task_states[projectno,3] = 0
                Task_states[projectno,4]= MPTD[projectno,1]+MPTD[projectno,2]+MPTD[projectno,3]+MPTD[projectno,4]-(pstatecode[projectno]-5)
                if size(MPTD,2) > 4
                    for task = 5:size(MPTD,2)
                        Task_states[projectno,task] = -1
                    end
                end
            elseif pstatecode[projectno]-1 <= MPTD[projectno,1]+MPTD[projectno,2]+MPTD[projectno,3]+MPTD[projectno,4]+MPTD[projectno,5]+5
                Task_states[projectno,1] = 0
                Task_states[projectno,2] = 0
                Task_states[projectno,3] = 0
                Task_states[projectno,4] = 0
                Task_states[projectno,5]= MPTD[projectno,1]+MPTD[projectno,2]+MPTD[projectno,3]+MPTD[projectno,4]+MPTD[projectno,5]-(pstatecode[projectno]-6)
            end
        end
        return Task_states
    end #change format of post decision state space (e.g., 7 => (0,5))  #task order needed
    function ResourceCheck(SS)
        #SS=[0 0 2;0 2 0]Test purpose
        Resource=Res1#resource type 1
        for  j = 1:size(SS,1)
            for i=1:size(SS,2)
                if SS[j,i] > 0  #&& i==1
                    Resource=Resource-MPRU[j,i]
                #elseif SS[j,i] >0 && i==2
                #    Resource=Resource-1
                end
            end
        end
        return  Resource
    end #returns free resources
    function greed_policy(T,RR,State_space)
        #State_space = [-1 -1 -1; -1 -1 -1; 0 0 3] #Test values
        #T=[1,1,1,1] #Test values
        #State_space = [0 0]
        #T = [3,1,1,1,1]
        #RR=3 #Test values
        #State space is suitable for diffirent project options and diffirent task number
        best_action = zeros(Int8, size(State_space,1),size(State_space,2))
        ###########################################################
        ##Here we are founding the doing nothing policy
        ##is it needed because we dont accepts doing nothing
        best_value = Bellman(T,State_space,best_action)#Value of do nothing
        #println("Policy finding",Bellman(T,State_space,best_action)," ",T," ",State_space," ",best_action)#Value of do nothing
        #println("Policy finding",Bellmantest(T,State_space,best_action)," ",T," ",State_space," ",best_action)#Value of do nothing
        locate=0 #counter for i and j locations
        ###from here we are finding the possible solituon space
        savei=[]
        savej=[]
        savet=[]
        #This loop locates the possible actions for the given state
        for  j = 1: size(State_space,1) #SS is my state space it has two dimention(projects and tasks)
            for i = 1: size(State_space,2)
                if State_space[j,i]==-1   #if activity was in-active
                    if i >1&&State_space[j,i-1]!=0
                        #This is task order control
                        #Need a task order control for more type of projects
                        #this part is only suitable for line projects
                    else
                        locate=locate+1
                        push!(savei,i)
                        push!(savej,j)
                        push!(savet,locate)
                    end
                end
            end
        end
        #Rusage=[] #remaining resources according to state t-1
        PSS = [] #for saving all possible action state spaces for Vt-1
        #This loop creates all possible actions policies, even though there are no feasible
        #and save only feasible ones according to given resource availabity
        for N in collect(combinations(savet))
            RTest=0 #resource requirments of all task will be active again
            MNew = zeros(Int8, size(State_space,1),size(State_space,2))
            for n in N
                RTest=RTest+MPRU[savej[n],savei[n]]
            end
            if RTest <= RR #if the total active project required less resource then available
                for n in N
                    MNew[savej[n],savei[n]]=1
                end
                push!(PSS,MNew)
            end
        end
        ###here we have found the solution  space
        ##now we are evoulating them
        #The code below prevent that doing nothing while whole resources are free.
        #it prevents selection of doing nothing if do some policy has same reward with it.
        #best_value has it value from calculation above
        best_actions = [] # holds the potantial best actions.
        #=tolerance =0.1^14#This the tolerance factor
        if RR == Res1 &&   PSS != [] #this if eleminates selection of do nothing
            best_action = PSS[1]
            best_value = Bellman(T,State_space,PSS[1])
            for a = 2 : size(PSS,1)
                    test_Value = Bellman(T,State_space,PSS[a])
                if test_Value>(best_value+tolerance)
                    best_value = test_Value
                    best_action = PSS[a]
                    push!(best_actions,PSS[a])
                elseif best_value>(best_value-tolerance)
                    best_value = max(best_value,test_Value)
                    push!(best_actions,PSS[a])
                end
            end
        else #this second part allows do nothing if there is no activity options or some resources busy
            for a in PSS
                #send the changed state space
                    test_Value = Bellman(T,State_space,a)
                    for a = 2 : size(PSS,1)
                            test_Value = Bellman(T,State_space,PSS[a])
                        if test_Value>(best_value+tolerance)
                            best_value = test_Value
                            best_action = PSS[a]
                            push!(best_actions,PSS[a])
                        elseif best_value>(best_value-tolerance)
                            best_value = max(best_value,test_Value)
                            push!(best_actions,PSS[a])
                        end
                    end
            end
        end=#
        if RR == Res1 &&   PSS != [] #this if eleminates selection of do nothing
            best_action = PSS[1]
            best_value = Bellman(T,State_space,PSS[1])
            for a = 2 : size(PSS,1)
                    test_Value = Bellman(T,State_space,PSS[a])
                if test_Value>best_value
                        best_value = test_Value
                    best_action = PSS[a]
                end
            end
        else #this second part allows do nothing if there is no activity options or some resources busy
            for a in PSS
                #send the changed state space
                    test_Value = Bellman(T,State_space,a)
                if test_Value>best_value
                        best_value = test_Value
                    best_action = a
                end
            end
        end
        #println(best_value)
        return best_action
    end#value iteration method of DP with max reward, #Still need a task ordering controller
    function Notgreed_policy(T,RR,State_space)
        #State_space = [-1 -1 -1; -1 -1 -1; 0 0 3] #Test values
        #T=[1,1,1,1] #Test values
        #State_space = [0 0]
        #T = [3,1,1,1,1]
        #RR=3 #Test values
        #State space is suitable for diffirent project options and diffirent task number
        best_action = zeros(Int8, size(State_space,1),size(State_space,2))
        ###########################################################
        ##Here we are founding the doing nothing policy
        ##is it needed because we dont accepts doing nothing
        best_value = Bellman(T,State_space,best_action)#Value of do nothing
        locate=0 #counter for i and j locations
        ###from here we are finding the possible solituon space
        savei=[]
        savej=[]
        savet=[]
        #This loop locates the possible actions for the given state
        for  j = 1: size(State_space,1) #SS is my state space it has two dimention(projects and tasks)
            for i = 1: size(State_space,2)
                if State_space[j,i]==-1   #if activity was in-active
                    if i >1&&State_space[j,i-1]!=0
                        #This is task order control
                        #Need a task order control for more type of projects
                        #this part is only suitable for line projects
                    else
                        locate=locate+1
                        push!(savei,i)
                        push!(savej,j)
                        push!(savet,locate)
                    end
                end
            end
        end
        #Rusage=[] #remaining resources according to state t-1
        PSS = [] #for saving all possible action state spaces for Vt-1
        #This loop creates all possible actions policies, even though there are no feasible
        #and save only feasible ones according to given resource availabity
        for N in collect(combinations(savet))
            RTest=0 #resource requirments of all task will be active again
            MNew = zeros(Int8, size(State_space,1),size(State_space,2))
            for n in N
                RTest=RTest+MPRU[savej[n],savei[n]]
            end
            if RTest <= RR #if the total active project required less resource then available
                for n in N
                    MNew[savej[n],savei[n]]=1
                end
                push!(PSS,MNew)
            end
        end
        ###here we have found the solution  space
        ##now we are evoulating them
        #The code below prevent that doing nothing while whole resources are free.
        #it prevents selection of doing nothing if do some policy has same reward with it.
        #best_value has it value from calculation above
        best_actions = [] # holds the potantial best actions.
        #=tolerance =0.1^14#This the tolerance factor
        if RR == Res1 &&   PSS != [] #this if eleminates selection of do nothing
            best_action = PSS[1]
            best_value = Bellman(T,State_space,PSS[1])
            for a = 2 : size(PSS,1)
                    test_Value = Bellman(T,State_space,PSS[a])
                if test_Value>(best_value+tolerance)
                    best_value = test_Value
                    best_action = PSS[a]
                    push!(best_actions,PSS[a])
                elseif best_value>(best_value-tolerance)
                    best_value = max(best_value,test_Value)
                    push!(best_actions,PSS[a])
                end
            end
        else #this second part allows do nothing if there is no activity options or some resources busy
            for a in PSS
                #send the changed state space
                    test_Value = Bellman(T,State_space,a)
                    for a = 2 : size(PSS,1)
                            test_Value = Bellman(T,State_space,PSS[a])
                        if test_Value>(best_value+tolerance)
                            best_value = test_Value
                            best_action = PSS[a]
                            push!(best_actions,PSS[a])
                        elseif best_value>(best_value-tolerance)
                            best_value = max(best_value,test_Value)
                            push!(best_actions,PSS[a])
                        end
                    end
            end
        end=#
        if PSS != [] #this if eleminates selection of do nothing
            best_action = PSS[1]
            best_value = Bellman(T,State_space,PSS[1])
            for a = 2 : size(PSS,1)
                    test_Value = Bellman(T,State_space,PSS[a])
                if test_Value<best_value
                        best_value = test_Value
                    best_action = PSS[a]
                end
            end
        end
        #println(best_value)
        return best_action
    end# value iteration methof of DP with min reward, Still need a task ordering controller
    #=function Bellman22(ttime,SS,a) #is value iteration
        #SS = [0 0;0 0]
        #a = [0 0 ;0 0 ]
        #ttime = [5,3,1,1,1]
        t = copy(ttime)# t is updated remaining due date
        #Recent Change #I moved the time iteration after the reward calculation
        #Time iteration #this time iteration effect for future state, not this state
        #we receive the reward full if there is 1 day left
        #this should stay before reward iteratiton since it overrides the due dates when project arrival
        for n = 1: size(PDD,1)#This is my timer
            if ttime[n] == 1#if this is last day of due date, next day it will be overdue
                t[n] = PDD[n]+1 # t is updated remaining due date
            elseif ttime[n] <= PDD[n]#if there are more time for due date, reduce one from remaining time
                t[n] = t[n]-1
            end
        end
        ##Reward iteration
        val =0.00000000 #dis is my Vt+1 value, its defined as a global in for this function
        for j = 1:size(PDD,1)
            i=size(SS,2)
            #this is Reward calculation############################
            #if second task is active and task proces time is one or remained processing time of last task is one
            if (SS[j,i] ==1&& MPTD[j,i]!=1)   || (a[j,i] > 0 && MPTD[j,i]==1)
                val=val+sum(reward,dims=2)[j]
                t[j] = PDD[j]+1 # if project will finish that turn reflesh its remaining due date.

                #this is tardiness cost calculation############################
                if  ttime[j]==PDD[j]+1 #if project is overdue, there will be a lateness cost(reward-cost)
                    val=val-Tardiness[j] # -1 tardiness cost for not finished projects
                end
            end
        end
        ######arrival process start here##################
        arrival = [] #Saving finished project in that state
        for  projectno = 1:size(SS,1) #for each project
            if SS[projectno,size(SS,2)] == 0 #if project is finished
                push!(arrival,projectno) #save as arrival expected here
            end
        end
        Exponential(2)
        1-cdf(Exponential(10),1)


        State_V_plus_one = transition(SS,a) #calculating next state for case no arrivals
        result=[] #NEXT STATE AFTER ARRIVALS OCCORS
        prob=[] #possibility distribution for each option of arrivals
        times=[] #remained due dates for  each option of arrivals
        push!(prob,(1-ArrivalProbabilty)^length(arrival)) #Saving the prossbilty of no project arrived that turn
        push!(result,conventor(State_V_plus_one)) #Saving case which no new project arrives.
        push!(times,t) #Remained due date matrix for no new project arrival case.
        for N in collect(combinations(arrival)) #finding all project arrival options
            TS = copy(State_V_plus_one)
            tm = copy(t)
            for nn in N #Creates remained due dates for all project arrival options.
                for nnn = 1: size(TS[nn,:],1)
                    TS[nn,nnn]=-1
                end
                tm[nn] = PDD[nn] #refless due date if a project arrives
            end
            push!(prob,ArrivalProbabilty^length(N)*(1-ArrivalProbabilty)^(length(arrival)-length(N))) #possibility distribution for each option of arrivals
            push!(result,conventor(TS)) #NEXT STATE AFTER ARRIVALS OCCORS
            push!(times,tm) #remained due dates for  each option of arrivals
        end
        for n = 1:length(times) #number of suitable tasks
            #global val
            p1d,p2d,p3d,p4d,p5d = times[n] #pt=project_1_due dates
            p1,p2,p3,p4,p5 = result[n] #project_1 tasks
            val = val + prob[n]*(discount*Value[p1,p1d,p2,p2d,p3,p3d,p4,p4d,p5,p5d]) #1 is Prob
            #Value[i,j,k,l,m] is value of State_V_plus_one, given state space Vt+1
            #val = SS-A = Vt
        end
        return val #VT value
    end=##it should be suitable for more task option too
    #=function Bellman33(ttime,SS,a) #is value iteration
        # 1, 1, 2, 4, 1, 1, 1, 1
        #SS = [0 0; 0 2]
        #conventor(SS)
        #a = [0 0; 0 0]
        #ttime =  [1, 11, 1, 1, 1]
        t = copy(ttime)# t is updated remaining due date
        #Recent Change #I moved the time iteration after the reward calculation
        #Time iteration #this time iteration effect for future state, not this state
        #we receive the reward full if there is 1 day left
        #this should stay before reward iteratiton since it overrides the due dates when project arrival
        for n = 1: size(PDD,1)#This is my timer
            if ttime[n] == 1#if this is last day of due date, next day it will be overdue
                t[n] = PDD[n]+1 # t is updated remaining due date
            elseif ttime[n] <= PDD[n]#if there are more time for due date, reduce one from remaining time
                t[n] = t[n]-1
            end
        end
        ##########TEST##################
        ##This produce the early start
        #we need to add arrival prob as well
        val=0.00
        result=[] #next state of early arrivals
        prob=[]#arrival and task durations probability
        Taskprob=[]#task durations probability
        Rval =[]#Reward value
        TaskCompletionP=[]#stochastic task completion project holder
        TaskCompletionT=[]#stochastic task completion task holder
        times=[] #remained due dates for  each option of arrivals
        ##Selecting of possible early finishes
        SomeProbabilty = 1.00
        for  projectno = 1:size(SS,1) #for each project
            for taskno = 1:size(SS,2)#for each task
                if SS[projectno,taskno] <= 3 &&  SS[projectno,taskno] > 1#if project is in last 3 round
                push!(TaskCompletionP,projectno) #save as projectno of earlyfinish
                push!(TaskCompletionT,taskno) #save as projectno of earlyfinish
                    #global# #SomeProbabilty (1-EarlyC[projectno,taskno,(SS[projectno,taskno]-1)]) #not Early completion prob
                end
            end
        end
        ##Here no early finish
        State_V_plus_one = transition(SS,a) #calculating next state for case no arrivals
        ####NO EARLY START OPTION
            #push!(result1,conventor(State_V_plus_one)) #Saving case which no new project arrives.
            for P in TaskCompletionP
                T = TaskCompletionT[findall(x->x==P , TaskCompletionP)[1]]#gives the referance of early completed task
                #println(P in N,T,TS)
            #global# #SomeProbabilty *=(1-disp(MPTD[P,T]-SS[P,T]+1,MPTD[P,T]-1))#not Early completion prob
            #global# SomeProbabilty *=(1-EarlyC[P,T,(SS[P,T]-1)]) #not Early completion prob
            end
            push!(Taskprob,SomeProbabilty) #SaveTaskprob
            #push!(TaskCompProb,SomeProbabilty) #Saving the prossbilty of non early finish
            #push!(times,t) #Remained due date matrix for no early finish
            prob,result,times = ProbabityOptions(prob,result,times,State_V_plus_one,SomeProbabilty,t)

            ##Reward iteration
            val =0.00000000 #dis is my Vt+1 value, its defined as a global in for this function
            for j = 1:size(PDD,1)
                i=size(SS,2)
                #this is Reward calculation############################
                #if second task is active and task proces time is one or remained processing time of last task is one
                if (SS[j,i] ==1&& MPTD[j,i]!=1)   || (a[j,i] > 0 && MPTD[j,i]==1)
                    val=val+sum(reward,dims=2)[j]
                    t[j] = PDD[j]+1 # if project will finish that turn reflesh its remaining due date.

                    #this is tardiness cost calculation############################
                    if  ttime[j]==PDD[j]+1 #if project is overdue, there will be a lateness cost(reward-cost)
                        val=val-Tardiness[j] # -1 tardiness cost for not finished projects
                    end
                end
            end
            push!(Rval,val) #Saverewardprob

        #####
        ####EARLY START OPTION
            for N in collect(combinations(TaskCompletionP)) #finding all task completion #Not suitable for every task matrix
                SomeProbabilty=1
                TS = copy(State_V_plus_one)
                tm = copy(t)
                val=Rval[1]
                for P in N #Creates remained due dates and rewards for early endings
                    T = TaskCompletionT[findall(x->x==P , TaskCompletionP)[1]] #gives the referance of early completed task
                    TS[P,T]=0#finishing the task early
                    if T == size(SS[P,:],1)##if this task is the last task of type nn project
                        #Reward iteration
                        val=val+sum(reward,dims=2)[P]
                        tm[P] = PDD[P]+1 #makes the early completed projects duration due
                        #this is tardiness cost calculation############################
                        if  ttime[P]==PDD[P]+1 #if project is overdue, there will be a lateness cost(reward-cost)
                            val=val-Tardiness[P] # -1 tardiness cost for not finished projects
                        end
                    end
                end
                push!(Rval,val) #Save reward options
                # this creates the probabilities of early completions
                for P in TaskCompletionP #it consider all early completable project not only early completed ones like N, So it calculates both early finishing and not finishing together
                    T = TaskCompletionT[findall(x->x==P , TaskCompletionP)[1]]#gives the referance of early completed task
                    #println(P in N,T,TS)
                    if P in N
                        #SomeProbabilty = SomeProbabilty*disp(MPTD[P,T]-SS[P,T]+1,MPTD[P,T]-1) #Early completion prob
                        SomeProbabilty = SomeProbabilty*EarlyC[P,T,(SS[P,T]-1)] #Early completion prob
                    else
                        #SomeProbabilty = SomeProbabilty*(1-disp(MPTD[P,T]-SS[P,T]+1,MPTD[P,T]-1)) #not Early completion prob
                        SomeProbabilty = SomeProbabilty*(1-EarlyC[P,T,(SS[P,T]-1)]) #not Early completion prob
                    end
                end
                push!(Taskprob,SomeProbabilty) #possibility distribution for each option of arrivals
                #push!(result1,conventor(TS)) #NEXT STATE early completion
                #push!(times,tm) #remained due dates for  early completion
                #global# prob,result,times = ProbabityOptions(prob,result,times,TS,SomeProbabilty,tm)
            end
        ##########TEST##################
        #sum(prob)
        #sum(Taskprob)
        val = 0.0
        for n = 1:length(Taskprob) #number of suitable tasks
            #global# val
            val = val + Taskprob[n]*Rval[n]
        end
        for n = 1:length(times) #number of suitable tasks
            #global# val
            p1d,p2d,p3d,p4d,p5d = times[n] #pt=project_1_due dates
            p1,p2,p3,p4,p5 = result[n] #project_1 tasks
            val = val + prob[n]*(discount*Value[p1,p1d,p2,p2d,p3,p3d,p4,p4d,p5,p5d]) #1 is Prob
            #Value[i,j,k,l,m] is value of State_V_plus_one, given state space Vt+1
            #val = SS-A = Vt
        end
        return val #VT value
    end=##it should be suitable for more task option too
    function Bellman(ttime,SS,a) #is value iteration
        #ttime,SS,a
        #[5, 6, 1, 1, 1] Int8[0 -1; 0 2] Int8[0 0; 0 0]
        #[5, 6, 1, 1, 1] Int8[0 -1; 0 2] Int8[0 0; 0 0]
        #1, 8, 5, 2, 1, 1, 1, 1
        #SS =  [-1 -1; 0 -1;-1 -1]
        #conventor(SS)
        #a = [1 0;0 0;0 0]
        #ttime =  [8, 2, 1, 1, 1]
        #policy[4, 7, 5, 1, 1, 1, 1, 1]
        ##########TEST##################
        ##This produce the early start
        #we need to add arrival prob as well
        val=0.00
        result=[] #next state of early arrivals
        prob=[]#arrival and task durations probability
        Taskprob=[]#task durations probability
        Rval =[]#Reward value
        TaskCompletionP=[]#stochastic task completion project holder
        TaskCompletionT=[]#stochastic task completion task holder
        times=[] #remained due dates for  each option of arrivals
        #println(ttime)
        ###Time FIX, THIS FIXES if there is remaining time for a completed project
        #for P = 1:size(PDD,1)
        #    if sum(SS,dims=2)[P]==0
        #        ttime[P]=PDD[P]+1
        #    end
        #end
        State_V_plus_one = copy(SS) #to prevent writing on current state space.

        for  j =1: size(SS,1) #SS is my state space it has two dimention
            for i =1: size(SS,2)
                #State + Action but not the next state
                #Next states are calculated beforehand
                if SS[j,i] == -1 && a[j,i] == 1 #This is
                    State_V_plus_one[j,i] = MPTD[j,i]#-1 #doest task process after action applied. It reduces at later in iteration
                end
            end
        end
        ##Transfer  State_V_plus_one to P1, P2, P3, P4, P5
        #println(SS,"   ",a)
        P1,P2,P3,P4,P5 = conventor2(State_V_plus_one)
        #P1,P2,P3,P4,P5 = 1,9,1,1,1
        #println(State_V_plus_one,"   ",P1,P2,P3,P4,P5,"   ",stateConventor2(P1,P2,P3,P4,P5))

        NState = copy(NEWState[P1,P2,P3,P4,P5]) #next states
        NProb = copy(NEWProb[P1,P2,P3,P4,P5]) #next state values
        NDD = copy(NEWDueDates[P1,P2,P3,P4,P5]) # nest state due dates
        TaskProb = copy(NEWTaskProb[P1,P2,P3,P4,P5]) #task probs
        TaskNDD = copy(NEWTaskDates[P1,P2,P3,P4,P5])#task due dates for reward calculation
        for NS = 1: size(NProb#=NEWProb[P1,P2,P3,P4,P5]=#,1) #NS is new states of SS
            ###TIME ITERATION
            #0 = no change, 1= iterate due date, 2= project finished, 3=project arrived#
            t = copy(ttime)# t is updated remaining due date
            val = 0.0
            for P = 1:size(NDD[NS],1)
                if NDD[NS][P] == 1
                    if ttime[P] == 1#if this is last day of due date, next day it will be overdue
                        t[P] = PDD[P]+1 # t is updated remaining due date
                    elseif ttime[P] <= PDD[P]#if there are more time for due date, reduce one from remaining time
                        t[P] = t[P]-1
                    end
                elseif NDD[NS][P] == 2  ##I can use them also in calculation
                    t[P] = PDD[P]+1 #  project finished
                    #val=val+sum(reward,dims=2)[P]
                    #this is tardiness cost calculation############################
                    #if  ttime[P]==PDD[P]+1 #if project is overdue, there will be a lateness cost(reward-cost)
                    #    val=val-Tardiness[P] # -1 tardiness cost for not finished projects
                    #end
                    #TaskProb = NEWProb[P1,P2,P3,P4,P5][NS]+NEWProb[P1,P2,P3,P4,P5][NS+1] #I have already have
                elseif NDD[NS][P] == 3
                    t[P] = PDD[P] #project arrived
                    #val=val+sum(reward,dims=2)[P]
                    ##bi tanesi yeterli sanirim
                    #this is tardiness cost calculation############################
                    #if  ttime[P]==PDD[P]+1 #if project is overdue, there will be a lateness cost(reward-cost)
                    #    val=val-Tardiness[P] # -1 tardiness cost for not finished projects
                    #end
                end
            end
            push!(times,t)
            push!(prob,NProb[NS])
            push!(result,NState[NS])
        end
        #sum(Taskprob)
        val = 0.0
        for NS = 1: size(TaskProb,1) #NS is new states of SS
            val = 0.0
            for P = 1:size(TaskNDD[NS],1)
                if TaskNDD[NS][P] == 2  ##I can use them also in calculation
                    val=val+sum(reward,dims=2)[P]
                    #this is tardiness cost calculation############################
                    if  ttime[P]==PDD[P]+1 #if project is overdue, there will be a lateness cost(reward-cost)
                        val=val-Tardiness[P] # -1 tardiness cost for not finished projects
                    end
                end
            end
            push!(Rval,val)
        end
        val = 0.0#it is required to clean value before value calculation otherwise it takes the previous val values
        for n = 1:length(TaskProb) #number of suitable tasks
            #global val
            val = val + TaskProb[n]*Rval[n]
        end

        for n = 1:size(times,1) #number of suitable tasks
            #global val
            #println("times",times)
            #println("result",result)
            p1d,p2d,p3d,p4d,p5d = times[n] #pt=project_1_due dates
            p1,p2,p3,p4,p5 = result[n] #project_1 tasks
             val = val + prob[n]*(discount*Value[p1,p1d,p2,p2d,p3,p3d,p4,p4d,p5,p5d]) #1 is Prob
            #Value[i,j,k,l,m] is value of State_V_plus_one, given state space Vt+1
            #val = SS-A = Vt

        end
        #println(val)
        return val #VT value
    end#bellmans value formula #it should be suitable for more task option too
    #@time Bellman()
    #=function ProbabityOptions(prob,result,times,IncomingState,SomeProbabilty,RD)
        IncomingState = [0 0;0 0]
        SomeProbabilty = 1.0
        RD = ttime = [2,5,1,1,1]
        prob=[]
        result=[]
        times=[]
        ######arrival process start here##################
        arrival = [] #Saving finished project in that state
        for  projectno = 1:size(IncomingState,1) #for each project
            if IncomingState[projectno,size(IncomingState,2)] == 0 #if project is finished
                push!(arrival,projectno) #save as arrival expected here
            end
        end
        IS = conventor(IncomingState) # Conventing for the suitable format
        AProbabilty = 1.00 #default arrival problity
        for P in arrival #This is the prob. of no arraval happening
            AProbabilty *= (1-ArrivalProb[P])
        end
        push!(prob,SomeProbabilty*AProbabilty) #Saving the prossbilty of no project arrived that turn
        push!(result,IS)# conventor(IncomingState)) #Saving case which no new project arrives.
        push!(times,RD) #Remained due date matrix for no new project arrival case.
        for Combination in collect(combinations(arrival)) #finding all project arrival options
            AlternateState = copy(IncomingState)
            AlternateTime = copy(RD)
            for P in Combination #Creates remained due dates for all project arrival options.
                for T = 1: size(AlternateState[P,:],1)
                    AlternateState[P,T]=-1
                end
                AlternateTime[P] = PDD[P] #refless due date if a project arrives
            end
            AS = conventor(AlternateState) #alternative state is not 5 project format AS is.
            #println(AlternateTime)
            AProbabilty = 1.00
            for P in arrival
                if P in Combination
                    AProbabilty *= [P]
                else
                    AProbabilty *= (1-ArrivalProb[P])
                end
            end
            push!(prob,SomeProbabilty*AProbabilty) #possibility distribution for each option of arrivals
            push!(result,AS) #conventor(AlternateState)) #NEXT STATE AFTER ARRIVALS OCCORS
            push!(times,AlternateTime) #remained due dates for  each option of arrivals
        end
        return prob, result, times
    end =# #used in old code
    function ProbabityOptions2(prob,result,times,IncomingState,SomeProbabilty,RD)
        #IncomingState = [0 -1;0 0]
        #SomeProbabilty = 0.22
        #RD = ttime = [1,2,0,0,0]
        #prob=[0.77]
        #result=[6,9,1,1,1]
        #times=[1,2,0,0,0]
        ######arrival process start here##################
        arrival = [] #Saving finished project in that state
        for  projectno = 1:size(IncomingState,1) #for each project
            if IncomingState[projectno,size(IncomingState,2)] == 0 #if project is finished
                push!(arrival,projectno) #save as arrival expected here
            end
        end
        IS = conventor(IncomingState) # Conventing for the suitable format
        AProbabilty = 1.00 #default arrival problity
        for P in arrival #This is the prob. of no arraval happening
            AProbabilty *= (1-ArrivalProb[P])
        end
        push!(prob,SomeProbabilty*AProbabilty) #Saving the prossbilty of no project arrived that turn
        push!(result,IS)# conventor(IncomingState)) #Saving case which no new project arrives.
        push!(times,RD) #Remained due date matrix for no new project arrival case.
        for Combination in collect(combinations(arrival)) #finding all project arrival options
            AlternateState = copy(IncomingState)
            AlternateTime = copy(RD)
            for P in Combination #Creates remained due dates for all project arrival options.
                for T = 1: size(AlternateState[P,:],1)
                    AlternateState[P,T]=-1
                end
                AlternateTime[P] = 3 #refless due date if a project arrives
            end
            AS = conventor(AlternateState) #alternative state is not 5 project format AS is.
            #println(AlternateTime)
            AProbabilty = 1.00
            for P in arrival
                if P in Combination
                    AProbabilty *= ArrivalProb[P]
                else
                    AProbabilty *= (1-ArrivalProb[P])
                end
            end
            push!(prob,SomeProbabilty*AProbabilty) #possibility distribution for each option of arrivals
            push!(result,AS) #conventor(AlternateState)) #NEXT STATE AFTER ARRIVALS OCCORS
            push!(times,AlternateTime) #remained due dates for  each option of arrivals
        end
        return prob, result, times
    end #finds future states,their probs and pdd changes,
    function transition(SS,AA)
        #SS=[0 0 2;0 -1 -1]#for test purpose
        #AA=[0 0 0;0 1 0]#for test purpose
        NewSS = copy(SS) #to prevent writing on current state space.
        for  j =1: size(SS,1) #SS is my state space it has two dimention
            for i =1: size(SS,2)
                if SS[j,i] == -1 && AA[j,i] == 1
                    NewSS[j,i] = MPTD[j,i]-1
                end
                if SS[j,i] > 0
                    #if n is acvive just found out its previous turn
                    NewSS[j,i]=NewSS[j,i]-1
                end
            end
        end    #A forward transition
        return NewSS
    end #deterministic next predecision state without uncertainty
    function conventor(Task_states) #state space conventor
        #Task_states = [3 -1; 0 0 ]#For test purpose
        pstatecode = fill(1,max_project)#
            for projectno = 1: size(PDD,1) #If project has no ongoing or finished task it is not started(1)
                if sum(Task_states,dims=2)[projectno] == -1*size(Task_states,2) && minimum(Task_states[projectno,:])==-1
                    pstatecode[projectno] = 1
                elseif sum(Task_states,dims=2)[projectno] == 0 && maximum(Task_states[projectno,:])==0
                    pstatecode[projectno] = 1+sum(MPTD,dims=2)[projectno]
                else
                    holdtasknumber,holdtaskvalue,holdfaketaskvalue =0,0,0 #dump files
                    for task in Task_states[projectno,:]
                        holdtasknumber+=1
                        holdtaskvalue+=MPTD[projectno,holdtasknumber]
                        if task >= sum(Task_states,dims=2)[projectno] &&
                            task < MPTD[projectno,holdtasknumber] &&
                            task != -1 &&
                             size(findall(x -> x > 0, Task_states[projectno,:]),1) < 2 &&#Only one task can be active
                             holdfaketaskvalue==0
                            pstatecode[projectno]=1+holdtaskvalue-Task_states[projectno,holdtasknumber]
                        elseif task != -1 && task != 0 ||  task >= MPTD[projectno,holdtasknumber] ||  task == 0 && holdfaketaskvalue!= 0
                            println("Alarm! Problem Related with unknown Task order in Conventor Funcktion Project no:",
                            projectno," task no: ",holdtasknumber)
                        end
                        holdfaketaskvalue=task
                    end
                end
            end
        return pstatecode[1],pstatecode[2],pstatecode[3],pstatecode[4],pstatecode[5]
    end#converts pre-decision task state formats ((0,5) => 7 )Not suitable for any task order
    function conventor2(Task_states) #state space conventor
        #Task_states = [1 -1; 0 0]#For test purpose
        pstatecode = fill(1,max_project)#
            for projectno = 1: size(PDD,1) #If project has no ongoing or finished task it is not started(1)
                if sum(Task_states,dims=2)[projectno] == -1*size(Task_states,2) && minimum(Task_states[projectno,:])==-1
                    pstatecode[projectno] = 1
                elseif sum(Task_states,dims=2)[projectno] == 0 && maximum(Task_states[projectno,:])==0
                    pstatecode[projectno] = 1+sum(MPTD,dims=2)[projectno]+size(MPTD,2)
                else
                    holdtasknumber,holdtaskvalue,holdfaketaskvalue =0,0,0 #dump files
                    for task in Task_states[projectno,:]
                        holdtasknumber+=1
                        holdtaskvalue+=MPTD[projectno,holdtasknumber]+1
                        if task >= sum(Task_states,dims=2)[projectno] &&
                            task <= MPTD[projectno,holdtasknumber] &&
                            task != -1 &&
                             size(findall(x -> x > 0, Task_states[projectno,:]),1) < 2 &&#Only one task can be active
                             holdfaketaskvalue==0
                            pstatecode[projectno]=1+holdtaskvalue-Task_states[projectno,holdtasknumber]
                        elseif task != -1 && task != 0 ||  task >= MPTD[projectno,holdtasknumber] ||  task == 0 && holdfaketaskvalue!= 0
                            println("Alarm! Problem Related with unknown Task order in Conventor Funcktion Project no:",
                            projectno," task no: ",holdtasknumber)
                        end
                        holdfaketaskvalue=task
                    end
                end
            end
        return pstatecode[1],pstatecode[2],pstatecode[3],pstatecode[4],pstatecode[5]
    end#converts post-decision task state formats ((0,5) => 7 )Not suitable for any task order
    function ActionConventor(Best_action)
        #Best_action=[1 0 0; 0 0 0;0 0 1;0 0 0]
        BA=0
        Tenholder = 1 #holding place of task in code
        for j = size(Best_action,1):-1:1 #backward for start from last project
            for i=size(Best_action,2):-1:1 # start from last task
                if Best_action[j,i] == 1
                    BA=BA + Tenholder
                end
                Tenholder=Tenholder*10
            end
        end
        return BA
    end #encodes the actions to single int code e.g., (1,0,1,0,0) => 10100
    function Print_results()
        #This is a senario simulation
        t1me = zeros(Int8,max_project)
        State_V_plus_one = zeros(Int8,max_project,size(MPTD,2))
        for projectno = 1:size(PDD,1)
            t1me[projectno] =PDD[projectno]
            for task = 1:size(State_V_plus_one[projectno,:],1)
                State_V_plus_one[projectno,task] =-1
            end
        end
        for projectno =  max_project:-1:(size(PDD,1)+1)
            t1me[projectno] =1 #1 has no meaning here
        end
        p1,p2,p3,p4,p5 = conventor(State_V_plus_one)
        ResourceTracker = []
        timer=[]
        ttt=1
        while State_V_plus_one != zeros(Int8,max_project,size(MPTD,2))
            #The code below gives the best policy for 2 project and fill the empty projects with 0
            Poly = PolicyConventor(policy[p1,t1me[1],p2,t1me[2],p3,t1me[3],p4,t1me[4],p5,t1me[5]]*(10^(size(MPTD,2)))^(max_project-size(PDD,1)))
            println(State_V_plus_one,"<==",Poly)
            push!(ResourceTracker,(3-ResourceCheck(State_V_plus_one))+(3-ResourceCheck(Poly)))
            push!(timer,ttt)
            ttt=ttt+1
            State_V_plus_one = transition(State_V_plus_one,Poly)
            p1,p2,p3,p4,p5 = conventor(State_V_plus_one)

            for a = 1:size(PDD,1)
                if t1me[a] > PDD[a] || t1me[a] == 1
                    t1me[a] = PDD[a]+1
                else
                    t1me[a] = t1me[a]-1
                end
            end
        end
        println(ResourceTracker)
        #plot(ResourceTracker,timer, color="red", linewidth=2.0)
    end #prints the applied policy from full stage to empty state without arrivals
    function PolicyConventor(PVal)
        #PVal=10000100
        projectstates = zeros(Int8,max_project,size(MPTD,2))
        control_number = 0
        project_number = 1
        for number = (max_project*size(MPTD,2)):-1:1
            control_number+=1
            if control_number > size(MPTD,2)
                control_number = 1
                project_number+=1
            end
            if PVal >= 10^(number-1)
                projectstates[project_number,control_number] = 1
                PVal-=10^(number-1)
            end
        end
        return projectstates
    end #encodes the int code actions to single e.g.,  10100 => (1,0,1,0,0)
    function cases()
        c,d,e,f,g,h,k,l=1,1,1,1,1,1,1,1
            for a = 1 : PDD[1]+1
                for b = 1:sum(MPTD,dims=2)[1]+1
                    if  a <= PDD[1]-(b-1)+DDSC[1,b] || a > PDD[1]
                        if size(PDD,1) > 1 #If more than 2 project exist
                            for c = 1 : PDD[2]+1
                                for d = 1:sum(MPTD,dims=2)[2]+1
                                    if  c <= PDD[2]-(d-1)+DDSC[2,d] || c > PDD[2]
                                        if size(PDD,1) > 2# If more than 3 project exist
                                            for e = 1 : PDD[3]+1
                                                for f = 1:sum(MPTD,dims=2)[3]+1
                                                    if e <= PDD[3]-(f-1)+DDSC[3,f] || e > PDD[3]
                                                        if size(PDD,1) > 3# If more than 4 project exist
                                                            for g = 1 : PDD[4]+1
                                                                for h = 1:sum(MPTD,dims=2)[4]+1
                                                                    if  g <= PDD[4]-(h-1)+DDSC[4,h] || g > PDD[4]
                                                                        if size(PDD,1) > 4# If more than 4 project exist
                                                                            for k = PDD[5]+1
                                                                                for l = 1:sum(MPTD,dims=2)[5]+1
                                                                                    if k <= PDD[5]-(l-1)+DDSC[5,l] || k > PDD[5]
                                                                                        DeltaFounder(a,b,c,d,e,f,g,h,k,l)
                                                                                    end
                                                                                end
                                                                            end
                                                                        else
                                                                            DeltaFounder(a,b,c,d,e,f,g,h,k,l)
                                                                        end# if only 4 project exist
                                                                    end
                                                                end
                                                            end
                                                        else
                                                            DeltaFounder(a,b,c,d,e,f,g,h,k,l)
                                                        end#if only 3 project exist
                                                    end
                                                end
                                            end
                                        else
                                            #println(a,b,c,d,e,f,g,h,k,l)
                                            DeltaFounder(a,b,c,d,e,f,g,h,k,l)
                                        end#if only 2 project exist
                                    end
                                end
                            end
                        else
                            DeltaFounder(a,b,c,d,e,f,g,h,k,l)
                        end #if only one project exist
                    end
                end
            end
    end #runs all suitable states of state space. suitable for any number of project with any number of task # not ready for any task order
    function DeltaFounder(p1t,p1,p2t,p2,p3t,p3,p4t,p4,p5t,p5)
        #p1t,p1,p2t,p2,p3t,p3,p4t,p4,p5t,p5 =1,1,1,8,1,1,1,1,1,1 #test values
        ###Here it find best action and updates state space values
        State_space = stateConventor(p1,p2,p3,p4,p5)#turn 1,1,1,1 to [0 0,0 0,0 0,0 0]
        RA = ResourceCheck(State_space) #RA=3-required resource for that stage
        ########NOTE calculating another bellman iteration after the greed_policy may not required that value can be get from greed_policy
        if RA >= 0 #if required resource is more than 3, that stage can not be exist so dont bother to calculate
            timee=[p1t,p2t,p3t,p4t,p5t]#holding time as an array to save writing space
            Best_action= greed_policy(timee,RA,State_space)
            #Best_action= Notgreed_policy(timee,RA,State_space)
            #Best_action= GeneticAlgorihm(timee,RA,State_space)
            #Best_action= PriortyRule(timee,RA,State_space)
            BA=ActionConventor(Best_action)#save the action [0 1,0 0,0 0,1 0] as 1000010 #no overide problem
            policy[p1,p1t,p2,p2t,p3,p3t,p4,p4t,p5,p5t]=BA#saving the best action for that state
            HoldValue[p1,p1t,p2,p2t,p3,p3t,p4,p4t,p5,p5t]=Bellman(timee,State_space,Best_action)#calculete the V_new (V_t)
        end
    end  #finds value and policy of given state #suitable for any number of project with any task #no change for task order needed
    function DeltaFounder2(p1t,p1,p2t,p2,p3t,p3,p4t,p4,p5t,p5)
        #p1t,p1,p2t,p2,p3t,p3,p4t,p4,p5t,p5 =1,2,1,1,1,2,1,1,1,1#test values

        ###Here it find best action and updates state space values

        State_space = stateConventor(p1,p2,p3,p4,p5)#turn 1,1,1,1 to [0 0,0 0,0 0,0 0]
        RA = ResourceCheck(State_space) #RA=3-required resource for that stage

        ########NOTE calculating another bellman iteration after the greed_policy may not required that value can be get from greed_policy
        if RA >= 0 #if required resource is more than 3, that stage can not be exist so dont bother to calculate
            timee=[p1t,p2t,p3t,p4t,p5t]#holding time as an array to save writing space
            Best_action= PolicyConventor(policy[p1,p1t,p2,p2t,p3,p3t,p4,p4t,p5,p5t]*(10^(size(MPTD,2)))^(max_project-size(PDD,1)))
            HoldValue[p1,p1t,p2,p2t,p3,p3t,p4,p4t,p5,p5t]=Bellman(timee,State_space,Best_action)#calculete the V_new (V_t)
        end
    end  #finds value of given state and policy ##suitable for any number of project with any task #no change for task order needed
    function main()
        stopper= 0.000001
        MinW = 0.0000
        t=0
        while true
            #a=0
            #=global=# t=t+1
            cases()
            global Value
            ValueDif = HoldValue[HoldValue.!=0]-Value[HoldValue.!=0]
            #global MinW
            MaxW = maximum(ValueDif)
            #println(HoldValue[3,3,2,9,1,1,1,1])

            if MinW > minimum(ValueDif) && t>2
               println("aha burda problem var")
               #println(findall(x->x==HoldValue[HoldValue.!=0][argmin(ValueDif)], HoldValue),
               #" - ",findall(x->x== Value[HoldValue.!=0][argmin(ValueDif)] , Value))
               println(HoldValue[HoldValue.!=0][argmin(ValueDif)]," - ",Value[HoldValue.!=0][argmin(ValueDif)])
               #break
            end

            #conventor2(stateConventor(4,4,1,1,1)
            #NEWState[5,5,1,1,1]
            #HoldValue[14, 1, 1, 1, 1, 1, 1, 1]
            #Value[14, 1, 1, 1, 1, 1, 1, 1]
            MinW = minimum(ValueDif)
            Delta=(MaxW-MinW)
            println(Delta,"=",MaxW, "-",MinW)
            Value=copy(HoldValue)
            Delta<=(stopper*MinW) &&break#||time()-timer<600
        end
        #Print_results()
    end #main code (old habit from C#) #suitable for any number of project with any task
    #=function simulation()
        iteration_number=1000000
        table = zeros(Float64,100,3)
        for ntimes = 100:-1:1
            RandomNumber = rand(1:99999999)
            table[ntimes,1] = RandomNumber
            srand(RandomNumber)
            #project_arrival = rand(1:5,100000)
            #project_arrival = digits(10000)
            #project_arrival = [0,0,1,0,0]
            t1me = zeros(Int8,max_project)
            projects = zeros(Int8,max_project)
            ###########I need to create beginning from seed#################
            State_V_plus_one = zeros(Int8,max_project,size(MPTD,2))
            for projectno = 1:size(PDD,1)
                t1me[projectno] =PDD[projectno] #
                State_V_plus_one[projectno,:] =-1
            end
            ###########I need to create beginning from seed#################
            for projectno =  max_project:-1:(size(PDD,1)+1)
                #this is for un used project slots
                t1me[projectno] =1 #1 has no meaning here
            end
            projects = conventor(State_V_plus_one)
            ResourceTracker = []
            timer=[]
            ttt=1

            Reward = 0
            variance = 0
            #seed_stepper = 0 #zeros(Int128,size(PDD,1))
            for n = 1:iteration_number
                Poly = PolicyConventor(policy[projects[1],t1me[1],projects[2],t1me[2],projects[3],t1me[3],projects[4],t1me[4],projects[5],t1me[5]]*(10^(size(MPTD,2)))^(max_project-size(PDD,1)))
                #println(State_V_plus_one,"<==",Poly)
                push!(ResourceTracker,(3-ResourceCheck(State_V_plus_one))+(3-ResourceCheck(Poly)))
                push!(timer,ttt)
                ttt=ttt+1
                ###this only iterates current project need project acceptance
                for projectno = 1:size(PDD,1)
                    value = 0
                    if (projects[projectno] == sum(MPTD,dims=2)[projectno] && MPTD[projectno,size(MPTD,2)] != 1) || (Poly[projectno,size(MPTD,2)]==1 && MPTD[projectno,size(MPTD,2)] == 1)
                        value += sum(reward,dims=2)[projectno]
                        if t1me[projectno] == PDD[projectno]+1
                            value -=Tardiness[projectno]
                        end
                        Reward += value
                        variance += value^2
                    end
                end
                State_V_plus_one = transition(State_V_plus_one,Poly)
                ####put project arraval here
                for projectno = 1:size(PDD,1)
                    if t1me[projectno] > PDD[projectno] || t1me[projectno] == 1
                        t1me[projectno] = PDD[projectno]+1
                    else
                        t1me[projectno] = t1me[projectno]-1
                    end
                    ##project arrival
                    if projects[projectno] == sum(MPTD,dims=2)[projectno]+1
                        ###this is arrival determener
                        #seed_stepper = seed_stepper+ 1
                        arrival = rand(1:5)#project_arrival[mod1(seed_stepper, size(project_arrival,1))]
                        if arrival == 1
                            State_V_plus_one[projectno,:] = -1### I need to update State_V_plus_one
                            t1me[projectno] = PDD[projectno]
                        end
                    end
                end
                projects = conventor(State_V_plus_one)
                ##earning reward if any project is finished
            end
            (Reward/iteration_number-1.115775 )/(√((variance-Reward)/iteration_number)/√iteration_number)
            table[ntimes,3] = (√((variance-Reward)/iteration_number)/√iteration_number)
            table[ntimes,2] = Reward/iteration_number
        end
        writecsv("mytable2.csv", table)
    end =##!!!!!!!not tested with stochastic task durations
    function save_data()
        save("policy.jld2", "policy", policy)
        #CSV.write("MaxW.csv",MaxW)
    end#Save policies and MaxW
    function load_data()
        policy = load("policy.jld2")["policy"]
        #MaxW = readcsv("MaxW.csv")[1]
    end#Read policies and MaxW
    #####HERE### GENETIC ALGORIHM CODES####
    #This use transition and resource check as weell##
    ################Change the GASCORE
    function GAScore(T,Priorty_space,RR,State_space)
        #State_space = [-1 -1 -1; -1 -1 -1; 0 0 3] #Test values
        #Priorty_space = [11531 21292 19821; 8348 25663 8255; 0 0 0]
        #RR=3 #Test values
        #T=[4,4,4]#Test values
        time = copy(T)
        Priorty = copy(Priorty_space)
        FreeResource = copy(RR)
        State = copy(State_space)
        Zero = zeros(Int16, size(State_space,1),size(State_space,2)) #holds emty action polices for state iteration


        #Activating the suitable tasks
        #################this do not allow not active a task when we have enough resources#####################
        ###########We may solve this by only try to schedulle first 3 in the priorties##################
        t = 0 #this is iteration number to calculate who much iteration required to complete the schedule.
        EarnRewards = 0 # This is for reward calculation for using it in policy ordering
        while State != Zero
            InsidePriorty = copy(Priorty) #To keep the priorty values of unscheduled tasks
            for a = 1 : size(State_space,1)
                for b = 1 : size(State_space,2)
                    if State[a,b] == -1#This keeps the maximum iteration limit
                        if FreeResource > 0 #There is no point to make a schedule if we do not have any free resources
                            #find the task with max Priorty
                            findmax(InsidePriorty)
                            #Pno, Tno =ind2sub(InsidePriorty,indmax(InsidePriorty))old code
                            Pno, Tno = argmax(InsidePriorty)[1], argmax(InsidePriorty)[2]#new#finds the location of max value
                            #Then send this task to resource control and task order control
                            #task order control
                            TEST = true
                            if Tno != 1
                                if State[Pno,Tno-1] != 0
                                    #Failed from task control
                                    InsidePriorty[Pno,Tno] = 0
                                    TEST = false
                                end #passed from task control
                            end #passed from task control
                            #resource control
                            if FreeResource < MPRU[Pno,Tno] && TEST != false
                                #Failed from resource control
                                InsidePriorty[Pno,Tno] = 0
                                TEST = false
                            end #passed from task control
                            #activing the selecting task
                            if TEST == true #schedule if the task did not fail in controils
                                Priorty[Pno,Tno] = 0
                                InsidePriorty[Pno,Tno] = 0
                                State[Pno,Tno] = MPTD[Pno,Tno]
                                FreeResource = FreeResource - MPRU[Pno,Tno]
                            end
                        end
                    end
                end
            end
            #Reward and Tardiness cost iteration
            for a = 1 : size(State_space,1)
                b = size(State_space,2)#Last task, Change this if you change the project network
                    if State[a,b]==1 #I receive the reward if the task will be completed end of this time unit
                        #Reward calculation
                        EarnRewards = EarnRewards+reward[a,b]
                        if time[a]-t<0#t is passed time units,
                            #tardiness cost calculation
                            EarnRewards = EarnRewards-Tardiness[a]
                        end
                    end
            end
            State = transition(State,Zero)
            FreeResource = ResourceCheck(State)
            t = 1 + t
            EarnRewards
        end
        ##Check the lenght of t to compare performance of the GA
        return t,EarnRewards
    end#Calculates the lenght of a schedule(schedile order)
    function GeneticAlgorihm(T,RR,State_space)
        #Here I am giving a scheduling order to each waiting task then I process them in this order
        #for example we have 3 free resources, first we check the task with biggest priorty point.
        #Then if this thas requires less than 3 resources and its predecissors are completed.
        #we can process that task at that time and if we still have available resources, we repeat this.
        #TEST VALIUES ###
        #State_space = [0 0;0 1]#[-1 -1 -1; -1 -1 -1; -1 -1 -1] #Test values
        #T=[5,6,1,1,1] #Test values
        #RR=2 #Test values
        ### TEST VALUES ~~~
        populationsize = 100 #Population Size
        Generationsize =100 #Generation Size
        Elitsize = 10 #number of best schedules which are saved without changing
        Mutation_rate = 50 #the percentage of how many new invidual created by crosover will get mutation
        Priorty_space = zeros(Int16, populationsize, size(State_space,1),size(State_space,2))#holds priorties of each task for scheduling order
        Schedile_scores= zeros(Int8, populationsize)#holds schedule scores
        Schedile_rewards= zeros(Int8, populationsize)#holds schedule rewards
        NewGeneration = zeros(Int16, populationsize, size(State_space,1),size(State_space,2))#holds priorties of each task for scheduling order
        NewGenerationScores = zeros(Int8, populationsize)#holds schedule scores
        control = false#controls if there is only one action available
            pop = 1
            for a = 1 : size(State_space,1)
                for b = 1 : size(State_space,2)
                    if State_space[a,b] == -1
                        Priorty_space[pop,a,b]= rand(1:30000)
                    end
                end
            end#this generates scheduling orders for waiting tasks
            control, best_action = aControl(Priorty_space[pop,:,:],RR,State_space)
            if control
                return best_action
            end
            Schedile_scores[pop],Schedile_rewards[pop] = GAScore(T,Priorty_space[pop,:,:],RR,State_space)
        for pop = 2 : populationsize
            for a = 1 : size(State_space,1)
                for b = 1 : size(State_space,2)
                    if State_space[a,b] == -1
                        Priorty_space[pop,a,b]= rand(1:30000)
                    end
                end
            end#this generates scheduling orders for waiting tasks
            Schedile_scores[pop],Schedile_rewards[pop] = GAScore(T,Priorty_space[pop,:,:],RR,State_space)
        end##Now I have 100 random schedule and their scores
        if maximum(Priorty_space) != 0
            for gen = 1 : Generationsize
                Priorty_space, Schedile_scores,Schedile_rewards = Ranking(Priorty_space, Schedile_scores, Schedile_rewards) # ranking the schedules
                for ind = 1:Elitsize
                    NewGeneration[ind,:,:] =  Priorty_space[ind,:,:]
                    NewGenerationScores[ind] = Schedile_scores[ind]
                end #Elitist selection
                for ind = Elitsize+1:populationsize
                    NewGeneration[ind,:,:] =  Crosover(Priorty_space,Mutation_rate)
                    Schedile_scores[ind],Schedile_rewards[ind] = GAScore(T,NewGeneration[ind,:,:],RR,State_space)
                    #mutation is included inside of the crosover code
                end #rest of the population is created by crosover
            end
        end
        Priorty_space, Schedile_scores,Schedile_rewards = Ranking(Priorty_space, Schedile_scores,Schedile_rewards) # ranking the schedules
        best_action = TurnScheduletoAction(Priorty_space[1,:,:],RR,State_space) #This is best solution found
        return best_action
    end #policy creator with GA
    function Ranking(Priorty_space, Schedile_scores, Schedile_rewards)
        #shellsort shorting from rosettacode.org
        #Schedile_scores = [17 , 15 , 10 ,15 , 12 , 12] #test values
        #Priorty_space = [11531 21292  19821; 8348 25663 8255; 1 2 3; 3 3 3;1 1 1;0 0 0] #test values
        #Schedile_rewards = [17 , 15 , 18 ,21 , 21 , 20] #test values
        #minimize time
        incr = div(length(Schedile_scores), 2)
        while incr > 0
            for i in incr+1:length(Schedile_scores)
                j = i
                tmp = Schedile_scores[i]
                tmp2 = Priorty_space[i,:,:]
                tmp3 = Schedile_rewards[i]
                while j > incr && Schedile_scores[j - incr] > tmp
                    Schedile_scores[j] = Schedile_scores[j-incr]
                    Schedile_rewards[j] = Schedile_rewards[j-incr]
                    Priorty_space[j,:,:] = Priorty_space[j-incr,:,:]
                    j -= incr
                end
                Schedile_scores[j] = tmp
                Schedile_rewards[j] = tmp3
                Priorty_space[j,:,:] = tmp2
            end
            if incr == 2
                incr = 1
            else
                incr = floor(Int, incr * 5.0 / 11)
            end
        end
        #maximize reward
        incr = div(length(Schedile_rewards), 2)
        while incr > 0
            for i in incr+1:length(Schedile_rewards)
                i=4
                j = i
                tmp = Schedile_rewards[i]
                tmp2 = Priorty_space[i,:,:]
                tmp3 = Schedile_scores[i]
                while j > incr && Schedile_rewards[j - incr] < tmp
                    Schedile_scores[j] = Schedile_scores[j-incr]
                    Schedile_rewards[j] = Schedile_rewards[j-incr]
                    Priorty_space[j,:,:] = Priorty_space[j-incr,:,:]
                    j -= incr
                end
                Schedile_scores[j] = tmp3
                Schedile_rewards[j] = tmp
                Priorty_space[j,:,:] = tmp2
            end
            if incr == 2
                incr = 1
            else
                incr = floor(Int, incr * 5.0 / 11)
            end
        end
        return Priorty_space, Schedile_scores,Schedile_rewards
    end#shorting the schedules using shellsort
    function Crosover(Priorty_space,Mutation_rate)
        Fno = rand(1:size(Priorty_space,1)) # number of father task # maybe size(,1) is better try tomorrow
        Mno = rand(1:size(Priorty_space,1)) # number of mother task
        #Father_score = Schedile_scores[Fno] #scores are not needed
        #Mother_score = Schedile_scores[Mno]
        Father_priorty = Priorty_space[Fno,:,:] #selecting candidates for Crosover
        Mother_priorty = Priorty_space[Mno,:,:]
        #Kid_priorty = zeros(Int16, size(State_space,1),size(State_space,2))
        Crossoverpointforprojects = rand(1:size(Mother_priorty,1))
        Crossoverpointfortasks = rand(1:size(Mother_priorty,2)) #creating a random crossover ponint

        Kid_priorty = copy(Mother_priorty)

        for b = Crossoverpointfortasks : size(Priorty_space,3)
            Kid_priorty[Crossoverpointforprojects,b] = Father_priorty[Crossoverpointforprojects,b]
        end
        if Crossoverpointforprojects != size(Priorty_space,2)
            for a = Crossoverpointforprojects+1 : size(Priorty_space,2)
                Kid_priorty[a,:] = Father_priorty[a,:]
            end#this generates
        end
        prob=rand(1:100)
        if prob <= Mutation_rate
            Kid_priorty = mutation(Kid_priorty)
        end
        return Kid_priorty
        #=
        private int[] CrossOver(int[][] genhavuz, double ortbit, double varyans, int[][] kisalt)
               {

                   // tek Crosover noktası koydum
                   int[] Baba = new int[Faaliyetsayısı];
                   int[] Anne = new int[Faaliyetsayısı];
                   int[] Cocuk = new int[Faaliyetsayısı];
                   int[] KroBaba = new int[Faaliyetsayısı];
                   int[] KroAnne = new int[Faaliyetsayısı];
                   int[] KroCocuk = new int[Faaliyetsayısı];
                   int uygun = 0;
                   int kuygun = 0;

                   //Random rastgele = new Random();

                   ####This prevent to selecting worst schedules.
                   for (int an = 0; an < Populasyon; an++)
                   {
                       if (kisalt[an][Faaliyetsayısı - 1] <= ortbit - (CrossOverOranı * varyans))
                       { kuygun++; }
                       if (kisalt[an][Faaliyetsayısı - 1] > ortbit + (CrossOverOranı * varyans))
                       { uygun++; }
                   }

                   int[] GenetikHavuz2 = new int[Faaliyetsayısı];

                   kuygun = 0;
                       int a = rastgele.Next(kuygun, Populasyon - uygun );
                       int b = rastgele.Next(kuygun, Populasyon - uygun);
                       Anne = genhavuz[b];
                       Baba = genhavuz[a];
                   for (int i = 0; i < Faaliyetsayısı; i++)
                   {
                       Cocuk[i] = -1;
                   }

                   int CrosOverPoint = rastgele.Next(1, Faaliyetsayısı);

                   Array.Copy(Baba, 0, Cocuk, 0, (CrosOverPoint));

                   int tut = 0;

                   int index = 0;
                   for (int an = 0; an < Faaliyetsayısı; an++)
                   {
                       index = 0;

                       index = Array.IndexOf(Cocuk, Anne[an]);
                       if(index  == -1)
                       {
                           Cocuk[CrosOverPoint  + tut] = Anne[an];
                           tut++;
                       }
                   }

                   return Cocuk;

               }

        return=#
    end
    function mutation(Kid_priorty)
        #Kid_priorty=[0 44343 0;454 0 0] test value
        Holdp = []
        Holdt = []
        for a = 1: size(Kid_priorty,1)
            for b = 1: size(Kid_priorty,2)
                if Kid_priorty[a,b] != 0
                    push!(Holdp,a)
                    push!(Holdt,b)
                end
            end
        end
        Random = rand(1:size(Holdp,1))
        Kid_priorty[Holdp[Random],Holdt[Random]] = rand(1:30000)

        return Kid_priorty
        #=private static int[] Mutasyon(int[] Anne)
            {
                int[] terssira = new int[Faaliyetsayısı];
                // mutasyon sadece süreyi küçültmeye çalışıyor

                int[] Cocuk = new int[Faaliyetsayısı];



                int tersayitut = 0;
                int say = 0;

                int MutasyonPoint = rastgele.Next(0, Faaliyetsayısı);

                for (int fa = 0; fa < Faaliyetsayısı; fa++)
                {
                    tersayitut = Anne[fa];
                    terssira[Anne[fa]] = fa;
                }

                if (Liste[Anne[MutasyonPoint]].Hof != null)
                {
                    say = 0;
                    foreach (Faaliyet faaliyet in Liste[Anne[MutasyonPoint]].Hof)
                    {
                        if (terssira[(Convert.ToInt32(faaliyet.Ad) - 1)] > say)
                            say = terssira[(Convert.ToInt32(faaliyet.Ad) - 1)];
                    }
                }
                int b = rastgele.Next(say + 1, MutasyonPoint + 1);
                Cocuk[b] = Anne[MutasyonPoint];
                for (int i = 0; i < b; i++)
                { Cocuk[i] = Anne[i]; }
                for (int i = b + 1; i < MutasyonPoint + 1; i++)
                { Cocuk[i] = Anne[i - 1]; }
                for (int i = MutasyonPoint + 1; i < Faaliyetsayısı; i++)
                { Cocuk[i] = Anne[i]; }


                return Cocuk;
            }
        return=#
    end
    function TurnScheduletoAction(Priorty_space,RR,State_space)
        #State_space = [-1 -1 -1; -1 -1 -1]#; 0 0 3] #Test values
        #RR=3 #Test values
        #Priorty_space = [11531 21292 19821; 8348 25663 8255]
        InsidePriorty = copy(Priorty_space)
        FreeResource = copy(RR)
        State = copy(State_space)
        Zero = zeros(Int16, size(State_space,1),size(State_space,2)) #holds emty action polices for state iteration
        Action = zeros(Int8, size(State_space,1),size(State_space,2))
            for a = 1 : size(State_space,1)
                for b = 1 : size(State_space,2)
                    if State[a,b] == -1
                        if FreeResource > 0 #There is no point to make a schedule if we do not have any free resources
                            #find the task with max Priorty
                            #findmax(InsidePriorty)
                            #Pno, Tno =ind2sub(InsidePriorty,indmax(InsidePriorty)) old code
                            Pno, Tno = argmax(InsidePriorty)[1], argmax(InsidePriorty)[2]#new#finds the location of max value
                            #Then send this task to resource control and task order control
                            #task order control
                            TEST = true
                            if Tno != 1
                                if State[Pno,Tno-1] != 0
                                    #Failed from task control
                                    InsidePriorty[Pno,Tno] = 0
                                    TEST = false
                                end #passed from task control
                            end #passed from task control
                            #resource control
                            if FreeResource < MPRU[Pno,Tno] && TEST != false
                                #Failed from resource control
                                InsidePriorty[Pno,Tno] = 0
                                TEST = false
                            end #passed from task control
                            #activing the selecting task
                            if TEST == true #schedule if the task did not fail in controils
                                InsidePriorty[Pno,Tno] = 0
                                Action[Pno,Tno] = 1
                                FreeResource = FreeResource - MPRU[Pno,Tno]
                            end
                        end
                    end
                end
            end
        return Action
    end#turning schedulis to one state action plan
    function aControl(Priorty_space,RR,State_space)
        #State_space = [-1 -1 -1; -1 -1 -1; 0 0 3] #Test values
        #Priorty_space = [11531 21292 19821; 8348 25663 8255; 0 0 0]
        #RR=3 #Test values
        #Activating the suitable tasks
        action = zeros(Int16, size(State_space,1),size(State_space,2))
        donotcontinue = true # do not do a GA
        ResouceCount = 0 # if all selectible task can be activate at the same time do not continue
        taskCount = 0 #if only one or less task can selected
        InsidePriorty = copy(Priorty_space) #To keep the priorty values of unscheduled tasks
            for a = 1 : size(State_space,1)
                for b = 1 : size(State_space,2)
                    if State_space[a,b] == -1#This keeps the maximum iteration limit
                        if RR > 0 #There is no point to make a schedule if we do not have any free resources
                            donotcontinue=false
                            #find the task with max Priorty
                            #findmax(InsidePriorty) #do I need this ?
                            #Pno, Tno =ind2sub(InsidePriorty,argmax(InsidePriorty)) this is old code not working
                            Pno, Tno = argmax(InsidePriorty)[1], argmax(InsidePriorty)[2]#new#finds the location of max value
                            #Then send this task to resource control and task order control
                            #task order control
                            TEST = true
                            if Tno != 1
                                if State_space[Pno,Tno-1] != 0
                                    #Failed from task control
                                    InsidePriorty[Pno,Tno] = 0
                                    TEST = false
                                end #passed from task control
                            end #passed from task control
                            #resource control
                            if RR < MPRU[Pno,Tno] && TEST != false
                                #Failed from resource control
                                InsidePriorty[Pno,Tno] = 0
                                TEST = false
                            end #passed from task control
                            #activing the selecting task
                            if TEST == true #schedule if the task did not fail in controils
                                taskCount  = taskCount+1
                                action[Pno,Tno] = 1
                                InsidePriorty[Pno,Tno] = 0
                                #State[Pno,Tno] = MPTD[Pno,Tno]
                                ResouceCount = ResouceCount + MPRU[Pno,Tno]
                            end
                        end
                    end
                end
            end
        ##Check the lenght of t to compare performance of the GA
        if ResouceCount <= RR
            donotcontinue = true
        end
        if taskCount <= 1
            donotcontinue = true
        end
        return donotcontinue, action #use this actuon if donotcontinue is true
    end#if there is only one action option for that state do not use GA
    #####End of genetic Algorihm codes########

    ########Priorty rule codes##########
    function PriortyRule(T,RR,State_space)
        #State_space = [0 0 0; 0 0 0]#; 0 0 3] #Test values
        #RR=3 #Test values


        Priorty_space = zeros(Int16, size(State_space,1),size(State_space,2))#holds priorties of each task for scheduling order
            for a = 1 : size(State_space,1)
                for b = 1 : size(State_space,2)
                    if State_space[a,b] == -1
                        Priorty_space[a,b]= MPTD[a,b]
                    end
                end
            end#this generates scheduling orders for waiting tasks
        FreeResource = copy(RR)
        State = copy(State_space)
        Time = copy(T)
        ###Copyed from TurnScheduletoAction
        Action = zeros(Int8, size(State_space,1),size(State_space,2))
            for a = 1 : size(State_space,1)
                for b = 1 : size(State_space,2)
                    if State[a,b] == -1
                        if FreeResource > 0 #There is no point to make a schedule if we do not have any free resources
                            #Put your priorty rule here##############
                            findmax(Priorty_space)
                            #Pno, Tno =ind2sub(Priorty_space,indmax(Priorty_space))old not working code
                            Pno, Tno = argmax(Priorty_space)[1], argmax(Priorty_space)[2]#new#finds the location of max value
                            ##########################
                            #Then send this task to resource control and task order control
                            #task order control
                            TEST = true
                            if Tno != 1
                                if State[Pno,Tno-1] != 0
                                    #Failed from task control
                                    Priorty_space[Pno,Tno] = 0
                                    TEST = false
                                end #passed from task control
                            end #passed from task control
                            #resource control
                            if FreeResource < MPRU[Pno,Tno] && TEST != false
                                #Failed from resource control
                                Priorty_space[Pno,Tno] = 0
                                TEST = false
                            end #passed from task control
                            #activing the selecting task
                            if TEST == true #schedule if the task did not fail in controils
                                Priorty_space[Pno,Tno] = 0
                                Action[Pno,Tno] = 1
                                FreeResource = FreeResource - MPRU[Pno,Tno]
                            end
                        end
                    end
                end
            end
        return Action
    end # creates policy with RBA
    ########################
    #Probscases()
