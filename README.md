# thesis_final


Allow me to describe the code. 

There are two main layers to the code: 

Outter layer takes x0 as input
-inner layer takes x0 as a constant, and it spits out a distance metric
based on distance metric, x0 is updated. 

All of "inner layer" code is in  b_household_computations.py. 
All of "outter layer" code is in smm_wrap.py


When I enable parallel processing for any part of the code (I think I might have a deadlock): 
      PROCESS TERMINATES WITH FOLLOWING MESSAGE:
      Process finished with exit code -1073741819 (0xC0000005)


When I disable all parallel processing for the code (I have no clue why):
      PROCESS TERMINATES WITH FOLLOWING MESSAGE:
      Process finished with exit code -1073740791 (0xC0000409)
      
      The exit code refers to STATUS_STACK_BUFFER_OVERRUN
