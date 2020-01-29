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

## Features

1. Multidimensional arrays
2. Support types:
   - bool_, uint8, int8, uint16, int16, float_
3. Slices are fully supported as views
4. Unary + binary operators
   - all but inplace operators
5. Functions and methods:
   - `max`, `min`, `all`, `any`, `sum`, `prod`
   - `minimum`, `maximum`
   - `concatenate`, `vstack`, `hstack`
   - `empty`, `full`, `zeros`, `ones`, `eye`
   - `flip`, `fliplr`, `flipud`
6. Supported float functions:
   - `acos`, `asin`, `atan`, `sin`, `cos`, `tan`
   - `acosh`, `asinh`, `atanh`, `sinh`, `cosh`, `tanh`
   - `ceil`, `floor`
   - `erf`, `erfc`
   - `exp`, `expm1`
   - `gamma`, `lgamma`
   - `log`, `log10`, `log2`
   - `sqrt`
7. Linear algebra
   - `dot`
   - `lu`, `solve`, `inv`, `det`
8. Extensive unittest based routines
   
## Missing features

1. Advanced indexing
2. uint32/int32/complex types
3. Inplace operators
4. @ operator 
5. Lots of linear algebra

## Compiling

Compile as a regular micropython external C module. At this time you will need to use the 
following fork and branch

https://github.com/ComplexArts/micropython/tree/equality

to fix some micropython issues have with equality.
See [this PR](https://github.com/micropython/micropython/pull/5479) for details on this issue.

