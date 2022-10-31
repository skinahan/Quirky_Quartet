# Q: Write a python program to calculate the specified entry in the fibonacci sequence using a dynamic programming approach. 
# The input to the program will be an integer n which represents the number of the fibonacci sequence to calculate and return.
# The output of the program will be the nth number in the fibonacci sequence.
# A: # A: The fibonacci sequence is defined such that each entry in the sequence is equal to the sum of the previous two entries.
# The basic solution for the fibonacci solution is to add the results obtained from calling the method recursively with n-1 and n-2 as inputs.
# By using an array structure to store the previous two fibonacci numbers, we can avoid calculating the same values more than once.
# A dynamic programming solution for the problem might be as follows:

def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1

    fib_array = [0,1]

    for i in range(2,n+1): # We need to calculate the fibonacci number for the input value of n. 
        # This means we need to calculate the fibonacci numbers from 2 up to and including n. 
        # The range function will generate a sequence of numbers from 2 up to but not including n+1. 

        next_fib = fib_array[i-2] + fib_array[i-1] # The next entry in the sequence is equal to the sum of the previous two entries. 

        fib_array.append(next_fib) # Add this value to our array of previously calculated values. 

    return next_fib