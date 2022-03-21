# IJOR-DSRCMPSP
This source code is used for computational analysis in Satic, U., Jacko, P., & Kirkbride, C. (2020). Performance evaluation of scheduling policies for the Dynamic and Stochastic Resource-Constrained Multi-Project Scheduling Problem. International Journal of Production Research. https://doi.org/10.1080/00207543.2020.1857450

This code is my first julia code. Thus the code might be written poorly and might be hard to work with. Please forgive me for this. Please read the usage notes for instructions about how to use the code. Please contact with u.satic@lancaster.ac.uk for your questions and help requests. 

# Usage

# 1) Running the code :
  First run the lines from 1-23. Then run the lines from 108 to 2027. Then run the lines from 24 to 104.
  
 # 2) Optaining results : 
 The code writes diffirences between maximum and minimum value increases of each iteration till the diffirence is smaller than the stopping criteria to the REPL console. The code also writes the running time to the REPL console when it completes. 
 
[Diffirences between maximum and minimum value increases] = [Maximum value increase]-[Minimum value increase]. 
In our paper, we used the average of final [Maximum value increase] and [Minimum value increase] as the long-term average prot per unit time.  
 


# Problem selection :
**Default problem :** The two projects with two tasks each problem is assined as the default problem. 
**Other pre-defined problems :**  To able to run test with other pre-defined problems, first, put one "#" sign to each line from 51 to 55. 
 Then remove one "#" sign from start of lines :
 from 44 to 48 for The three projects with two tasks each problem or 
 from 37 to 41 for The two projects with three tasks each problem or 
 from 31 to 35 for The four projects with two tasks each problem.
 **Creating costum problems :** Due to computational limits of dynamic programming, most problem might not work. Thus we do not suggest changing the test problem. Setting of default problem can be change for costumisation. 
MPTD =Int8[X Y;Z T], is used for (expected) task durations. 
MPRU =Int8[X Y;Z T], is used for resource usages, 
PDD= Int8[A,B], is used for project's due dates,
Tardiness=Int8[A,B], is used for project's tardiness cost
reward=Int8[X Y;Z T],is used for task's completion reward
    A is the first project type, B is the second project type
    X is first project type's first task. Y is first project type's second task. Z is second project type's first task. T is second project type's second task.
    
# Deterministic or stochastic task duration option selection :
                                    
