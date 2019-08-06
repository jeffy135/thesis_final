# thesis_final


Allow me to describe the code. 

There are two main layers to the code: 


+----------------------------------+---------+------------------------+----------------+
|               Col1               |  Col2   |          Col3          | Numeric Column |
+----------------------------------+---------+------------------------+----------------+
| Value 1                          | Value 2 | 123                    |           10.0 |
| Separate                         | cols    | with a tab or 4 spaces |       -2,027.1 |
| This is a row with only one cell |         |                        |                |
+----------------------------------+---------+------------------------+----------------+

|------------ outter layer takes x0 as input-------------------------|
|                                                                    |
|        _____________________________________________________       |
|        |        inner layer takes x0 as a constant         |       |
|        |       all of the inner layer code is in           |       |
|        |         b_household_computations.py.              |       |
|        |       this inner code will spit out a             |       |
|        |               distance metric.                    |       |
|        |___________________________________________________|       |
|                                                                    |
|-- the outter layer takes the above distance metric and update x0 --|


When I enable parallel processing for any part of the code (I think I might have a deadlock): 
      PROCESS TERMINATES WITH FOLLOWING MESSAGE:
      Process finished with exit code -1073741819 (0xC0000005)


When I disable all parallel processing for the code (I have no clue why):
      PROCESS TERMINATES WITH FOLLOWING MESSAGE:
      Process finished with exit code -1073740791 (0xC0000409)
      
      The exit code refers to STATUS_STACK_BUFFER_OVERRUN
