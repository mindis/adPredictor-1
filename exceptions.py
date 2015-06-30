__author__ = 'michaelpearmain'

'''
We provide a single file for exception handling for 2 reasons:

Often times a module/program will need to import from one sub-module to get a function that imports and makes use of the
code raising an exception.
To trap the exception with granularity, we need to import both the module and the module defining the exception.
This sort of derivative import requirement is the first step towards a convoluted web of imports within your package.

The more times you execute this pattern, the more interdependent and error-prone your package becomes.
Over time as the number of exceptions increases, it becomes more and more difficult to find all of the exceptions a
package is capable of raising.

Defining all exceptions in a single module provides one convenient place where a programmer can inspect to determine the
full surface-area of potential error conditions raised by your package.
'''

class OnlineLearningException(Exception):
    '''root for OnlineLearning Exceptions, only used to except any OnlineLearning error, never raised'''
    pass