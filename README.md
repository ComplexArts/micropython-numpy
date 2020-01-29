# micropython-numpy

`micropython-numpy` is a numpy-like array manipulation library for micropython.

The module is written in C and aims to provide as much compatibility as possible with 
select core numpy functionality. 

This beginnings of this project were heavily inspired by
the [ulab project](https://github.com/v923z/micropython-ulab).
In its current version however, virtually all functionality has been redesigned 
from the ground up to handle multidimensional arrays and array views.

Documentation is very lacking at the moment but a user familiar with numpy should be 
able to infer the syntax from the current numpy documentation.

## Compatibility

One major difference is that `shape` and `ndim` and `size` are not properties but methods. 
The reason for that is discussed [here](https://forum.micropython.org/viewtopic.php?t=2412).

In order to make your code run on both the micropython as well as the regular python 
numpy use the functions `np.shape`, `np.ndim`, and `np.shape`, instead, which are present
in both versions. For example, instead of

    a = np.array([1,2,3])
    shape = a.shape

use 

    a = np.array([1,2,3])
    shape = np.shape(a)

The property `a.shape` works only on regular numpy and the method `a.shape()` works 
only on micropython.
   