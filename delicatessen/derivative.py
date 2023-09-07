import numpy
import numpy as np
import scipy as sp
from scipy.stats import norm


def auto_differentiation(xk, f):
    r"""Forward-mode automatic differentiation. Automatic differentiation offers a way to compute the exact derivative,
    rather than numerically approximated (i.e., the central difference method). Automatic differentiation iteratively
    applies the chain rule through recursive calls to evaluate the derivative.

    Note
    ----
    This functionality is only intended for use behind the scenes in ``delicatessen``. Automatic differentiation is
    implemented from scratch to avoid additional dependencies.


    This is accomplished by the ``PrimalTangentPairs`` class, which is a special data type in ``delicatessen`` that
    stores pairs of the original evaluation and the corresponding derivative for a variety of different mathematical
    operations. This is what allows for the exact derivative calculation. The ``auto_differentiation`` function is
    a wrapper to access and use this class object as it is intended for derivative computations.

    Parameters
    ----------
    xk: ndarray, list, shape (n, )
        Point(s) or coordinate vector to evaluate the gradient at.
    f: callable
        Function of which to estimate the gradient of.

    Returns
    -------
    numpy.array :
        Corresponding array of the pairwise derivatives for all different input x values.

    Examples
    --------
    Loading necessary functions

    >>> import numpy as np
    >>> from delicatessen.derivative import auto_differentiation

    To illustrate use, we will compute the derivative of the following function

    .. math::

        f(x) = x^2 - x^1 + sin(x + \sqrt{x})

    >>> def f(x):
    >>>     return x**2 - x + np.sin(x + np.sqrt(x))

    If you work out the deriative by-hand, you will end up with the following

    .. math::

        2x - 1 + \left( \frac{1}{2 \sqrt{x}} + 1 \right) \cos(x + \sqrt{x})

    Instead, we can use automatic differentiation to evaluate the derivative at a specific point. Here, we will
    evaluate the derivative at :math:`x=1`

    >>> dy = auto_differentiation(xk=[1, ], f=f)

    which returns ``0.3757795``. This is the same as if you plugged in :math:`x=1` into the previous equation.

    Note
    ----
    If a derivative is not defined, then the function will return a ``NaN``.


    The derivative of a function with multiple inputs and multiple outputs can also be evaluated. Consider the following
    example with three inputs and two outputs

    >>> def f(x):
    >>>     return [x[0]**2 - x[1], np.sin(np.sqrt(x[1]) + x[2]) + x[2]*(x[1]**2)]

    >>> dy = auto_differentiation(xk=[0.7, 1.2, -0.9], f=f)

    which will return a 2-by-3 array of all the x-y pair derivatives at the given values. Here, the rows correspond to
    the output and the columns correspond to the inputs.

    References
    ----------
    Baydin AG, Pearlmutter BA, Radul AA, & Siskind JM. (2018). Automatic differentiation in machine learning: a survey.
    *Journal of Marchine Learning Research*, 18, 1-43.

    Rall LB & Corliss GF. (1996). An introduction to automatic differentiation. Computational Differentiation:
    Techniques, Applications, and Tools, 89, 1-18.
    """
    # Meta-information about function and inputs
    xshape = len(xk)                                                   # The number of inputs into the function

    # Set up Pair objects for evaluating gradient of function
    pairs_for_gradient = []                                            # Storage for the pairs to provide function
    for i in range(xshape):                                            # For each of the inputs
        partial = np.zeros_like(xk)                                    # ... generate array of that size of all zeroes
        partial[i] = 1                                                 # ... replace 0 with 1 for current index
        pairs_for_gradient.append(PrimalTangentPairs(xk[i], partial))  # ... then store as a primal,tangent pair
    x_to_eval = np.asarray(pairs_for_gradient)                         # Convert from list to NumPy array

    # Internal function to handle some specific exceptions when computing derivatives
    def f_deny_bool(function, x):
        """This internal function re-evaluates the input function with an additional operator. Namely zero is added
        to the function. This causes no difference in the primal or tangent pair. The purpose of this function is to
        work around a specific behavior of ``np.where(...)``.

        The function ``np.where`` can a weird call order to the standard Python operators. It calls ``__bool__`` as the
        final operation. This means that ``PrimalTangentPairs`` can only return a ``bool`` for ``np.where`` calls.
        Consider the following example

        .. code::

            np.where(x[0] >= 1, x[1]**2, 0)


        If ``x[0] >= 1`` then we should find the derivative of ``x[1]**2``. For ``np.where``, ``x[1]**2`` is evaluated
        before ``x[0] >=1``. As a result of the boolean piece being the last part evaluated, resulting in an empty
        array.

        By adding zero, we ensure that the final operation applied is an addition, and thus an float is returned
        after the boolean is evaluated. Essentially, we deny the call to ``__bool__`` being the last operation.
        """
        evalued = function(x)              # Short-name for evaluated function

        # Handling whether a single input (non-tuple like SciPy) is provided
        if isinstance(evalued, PrimalTangentPairs):     # If not a tuple, then gives back the Pair object
            eval_no_bool_end = evalued + 0              # ... then just add zero in this case
        else:                                           # Otherwise,
            eval_no_bool_end = []                       # ... empty list for storage
            for e in evalued:                           # ... for each evaluation in the function
                eval_no_bool_end.append(e+0)            # ... adding zero in a loop

        return eval_no_bool_end            # Return the evaluation with the final addition

    # Evaluating the function for the primal,tangent pairs
    evaluated_pair = f_deny_bool(function=f,                           # Evaluate the primal,tangent pair with function
                                 x=x_to_eval)                          # ... at the given values

    # Processing function output into gradient value or matrix
    evaluated_gradient = []                                            # List storage for the computed gradient
    if isinstance(evaluated_pair, PrimalTangentPairs):                 # Handle case where only single input
        evaluated_gradient.append([evaluated_pair.tangent, ])          # ... evaluate tangent and put list in list
    else:                                                              # In all other cases
        for pair in evaluated_pair:                                    # ... for each evaluated primal, tangent pair
            if isinstance(pair, PrimalTangentPairs):                   # ... if that pair is the key type
                evaluated_gradient.append(pair.tangent)                # ... then give back the derivative
            else:                                                      # ... otherwise row has no xk operations
                empty_array = np.array([0 for j in range(xshape)])     # ... create array of all zeroes
                evaluated_gradient.append(empty_array)                 # ... so derivative is always zero

    # Return evaluated gradient as a NumPy array
    if len(evaluated_gradient) == 1:                                   # If only consists of 1 item
        return np.asarray(evaluated_gradient[0])                       # ... return that item as NumPy array
    else:                                                              # Otherwise
        return np.asarray(evaluated_gradient)                          # ... return is as an array


class PrimalTangentPairs:
    """Unique class object for automatic differentiation. This class divides the inputs into 'primal' and 'tangent'
    pairs. The derivative is computed via operations on the pairs, which are then recurvsively called by this class.
    This process allows use to successively apply the chain rule.

    Note
    ----
    This class only handles how the data is setup (i.e., it does not automatically compute a derivative. The function
    for the derivative computation then takes this as input.


    This data class is only meant to interact with ``auto_differentiation`` and should not be used elsewhere.

    Parameters
    ----------
    primal :
        The x values for which the derivative computation is desired. Must be the same length as ``tangent``.
    tangent :
        Indicator for the location at which the derivatives is desired. Must be the same length as ``primal``.
    """
    # Some internal notes on handling ndarray's in this class (because there are some tricky issues). First, we do
    #   not want to override the priority of the NumPy operators (in fact we rely on it to take precedence). This is
    #   extremely important when trying to mix operators together that start with a numpy.ndarray, as we want
    #   everything to go to element-wise operations. So, we must not do something like the following
    # __array_priority__ = 2
    # Instead, we have modified all the operations that include `other` to check if the input is a np.ndarray. If it is
    #   (i.e., the np.ndarray is the second object in x [] y) then we manually reverse the operations so that the
    #   np.ndarray leads (i.e., y [] x). This gets NumPy to decompose everything correctly to element-wise operations.
    #   The downside is that this introduces floating point errors for powers, when it is being raised to a np.ndarray
    #   power and everything is integers. This floating point error was determined to be acceptable to get everything
    #   working for the automatic differentation.

    def __init__(self, primal, tangent):
        # Processing of the inputs into the class, both initial and recursive calls
        if isinstance(primal, PrimalTangentPairs):    # If given a PrimalTangentPair input
            if isinstance(primal.primal, np.ndarray):
                raise ValueError("... order of operations issue... I am getting an array an input")
            self.primal = primal.primal               # ... extract the primal element from input
        else:                                         # Else
            if isinstance(primal, np.ndarray):
                raise ValueError("... order of operations issue... I am getting an array an input")
            self.primal = primal                      # ... directly save as new primal
        self.tangent = tangent                        # Store the tangent

    # Basic operators for the class object

    def __str__(self):
        # Conversion to string just to in case it gets called somehow
        return f"PrimalTangentPairs({self.primal}, {self.tangent})"

    def __bool__(self):
        # To get np.where working properly, I need to have the internal boolean function for this class return
        #   only the primal part. This is a bool object type, which is what is expected and has np.where operate as
        #   expected. This only seems to be called for np.where and not the other operators (they work directly).
        return bool(self.primal)

    def transpose(self):
        # Transpose operator is special
        storage = []  # Create empty storage for the output

        if self.tangent.ndim == 1:  # When tangent is of 1 dimension
            for p in self.primal:  # ... then only the primal's are a stack
                storage.append(PrimalTangentPairs(p,  # ... so we reshape the primal stack
                                                  self.tangent))  # ... and keep all the corresponding tangent's
        else:  # Otherwise, tangent is also 2 dimensions
            for p, t in zip(self.primal, self.tangent):  # ... so both the primal and tangents
                storage.append(PrimalTangentPairs(p, t))  # ... need to be re-paired

        # Return the transposed array of the PrimalTangentPairs
        return np.array(storage)

    @property
    def T(self):
        # Giving the transpose() property the .T short-cut, like NumPy has (to be compatible with NumPy code)
        return self.transpose()

    # Equality operators for the class

    def __le__(self, other):
        # Less than or equal to operator
        if isinstance(other, PrimalTangentPairs):                  # If the input is a PrimalTangentPairs
            comparison = (self.primal <= other.primal)             # ... compare the primals
        else:                                                      # Otherwise
            comparison = (self.primal <= other)                    # ... directly compare primal with other
        return PrimalTangentPairs(comparison, 0)                   # Return evaluated equality

    def __lt__(self, other):
        # Less than operator
        if isinstance(other, PrimalTangentPairs):                  # If the input is a PrimalTangentPairs
            comparison = (self.primal < other.primal)              # ... compare the primals
        else:                                                      # Otherwise
            comparison = (self.primal < other)                     # ... directly compare primal with other
        return PrimalTangentPairs(comparison, 0)                   # Return evaluated equality

    def __ge__(self, other):
        # Greater than or equal to operator
        if isinstance(other, PrimalTangentPairs):                  # If the input is a PrimalTangentPairs
            comparison = (self.primal >= other.primal)             # ... compare the primals
        else:                                                      # Otherwise
            comparison = (self.primal >= other)                    # ... directly compare primal with other
        return PrimalTangentPairs(comparison, 0)                   # Return evaluated equality

    def __gt__(self, other):
        # Greater than operator
        if isinstance(other, PrimalTangentPairs):                  # If the input is a PrimalTangentPairs
            comparison = (self.primal > other.primal)              # ... compare the primals
        else:                                                      # Otherwise
            comparison = (self.primal > other)                     # ... directly compare primal with other
        return PrimalTangentPairs(comparison, 0)                   # Return evaluated equality

    def __eq__(self, other):
        # Equality operator
        if isinstance(other, PrimalTangentPairs):                  # If the input is a PrimalTangentPairs
            comparison = (self.primal == other.primal)             # ... compare the primals
        else:                                                      # Otherwise
            comparison = (self.primal == other)                    # ... directly compare primal with other
        return PrimalTangentPairs(comparison, 0)                   # Return evaluated equality

    def __ne__(self, other):
        # Inequality operator
        if isinstance(other, PrimalTangentPairs):                  # If the input is a PrimalTangentPairs
            comparison = (self.primal != other.primal)             # ... compare the primals
        else:                                                      # Otherwise
            comparison = (self.primal != other)                    # ... directly compare primal with other
        return PrimalTangentPairs(comparison, 0)                   # Return evaluated equality

    # Elementary mathematical operations

    def __neg__(self):
        # Negation operator
        return PrimalTangentPairs(-self.primal,                    # Negation is simply negation
                                  -self.tangent)                   # ... and same for derivative

    def __add__(self, other):
        # Addition
        if isinstance(other, PrimalTangentPairs):                     # When adding PrimalTangentPairs
            return PrimalTangentPairs(self.primal + other.primal,     # ... add primals together
                                      self.tangent + other.tangent)   # ... and add tangents together
        elif isinstance(other, np.ndarray):
            return other + self
        else:                                                         # Otherwise
            return PrimalTangentPairs(self.primal + other,            # ... add the primal and other
                                      self.tangent)                   # ... tangent does not change by constants

    def __radd__(self, other):
        # Addition, reversed
        return self.__add__(other)                                    # Reversed addition (order matters for process)

    def __sub__(self, other):
        # Subtraction
        if isinstance(other, PrimalTangentPairs):                     # When subtracting PrimalTangentPairs
            return PrimalTangentPairs(self.primal - other.primal,     # ... minus primals together
                                      self.tangent - other.tangent)   # ... and minus tangets together
        elif isinstance(other, np.ndarray):
            return -1*other + self
        else:                                                         # Otherwise
            return PrimalTangentPairs(self.primal - other,            # ... minus primal and other
                                      self.tangent)                   # ... tangent does not change by constants

    def __rsub__(self, other):
        # Subtraction, reversed
        return PrimalTangentPairs(other - self.primal,                # Subtracting in reverse order (lead constant)
                                  -1 * self.tangent)                  # ... simply multiply tangent by -1

    def __mul__(self, other):
        # Multiplication
        if isinstance(other, PrimalTangentPairs):                     # When multiplying PrimalTangentPairs
            return PrimalTangentPairs(self.primal * other.primal,     # ... multiply primals
                                      self.tangent*other.primal + self.primal*other.tangent)
        elif isinstance(other, np.ndarray):                           # When comparing with np.ndarray
            return other * self                                       # ... reverse the order to make operations work
        else:                                                         # Otherwise
            return PrimalTangentPairs(self.primal * other,            # ... multiply primal and constant
                                      self.tangent * other)           # ... multiply primal and constant

    def __rmul__(self, other):
        # Multiplication, reversed
        return PrimalTangentPairs(self.primal * other,                # Reversed order can evaluate the same
                                  self.tangent * other)               # ... for primal and tangent pairs

    def __truediv__(self, other):
        # Division
        if isinstance(other, PrimalTangentPairs):                     # When dividing PrimalTangentPairs
            return PrimalTangentPairs(self.primal / other.primal,     # ... divide primals
                                      (self.tangent * other.primal - self.primal * other.tangent) / (other.primal ** 2))
        elif isinstance(other, np.ndarray):                           # When comparing with np.ndarray
            return (1/other) * self                                   # ... reverse the order to make operations work
        else:                                                         # Otherwise
            return PrimalTangentPairs(self.primal / other,            # ... divide primal and constant
                                      self.tangent / other)           # ... divide tangent and constant

    def __rtruediv__(self, other):
        # Division, reversed
        return PrimalTangentPairs(other / self.primal,                # Reversed form of division
                                  (0 * self.primal - self.tangent * other) / (self.primal ** 2))

    def __pow__(self, other):
        # Power
        if isinstance(other, PrimalTangentPairs):                                    # For PrimalTangentPairs to powers
            fgx = other.primal * self.primal ** (other.primal - 1) * self.tangent    # ... initial piece of rule
            gx = self.primal ** other.primal * np.log(self.primal) * other.tangent   # ... secondary piece of rule
            return PrimalTangentPairs(self.primal ** other.primal,                   # ... raise primal to primal
                                      fgx + gx)                                      # ... apply the power rule
        elif isinstance(other, np.ndarray):
            if self.primal >= 0:
                return np.exp(other * np.log(self))
            else:
                return np.where(other % 2 == 1,
                                -np.exp(other * np.log(-self)),
                                np.exp(other * np.log(-self)))
        else:
            return PrimalTangentPairs(self.primal ** other,                          # Otherwise compute directly
                                      other * self.primal ** (other - 1) * self.tangent)

    def __rpow__(self, other):
        # Power, reversed
        return PrimalTangentPairs(other ** self.primal,                                 # Constant to primal
                                  self.tangent * other ** self.primal * np.log(other))  # ... then power rule

    # Operators that sometimes have undefined derivatives

    def __abs__(self):
        # Absolute value
        if self.primal > 0:                                                            # Absolute value for positive
            return PrimalTangentPairs(1 * self.primal, self.tangent)                   # ... primal, tangent
        elif self.primal < 0:                                                          # Absolute value for negative
            return PrimalTangentPairs(-1 * self.primal, -1 * self.tangent)             # ... -primal, -tangent
        else:                                                                          # Absolute value at zero
            return PrimalTangentPairs(0 * self.primal, np.nan * self.tangent)          # ... zero, undefined

    def __floor__(self):
        # Floor
        if self.primal % 1 == 0:                                                       # Floor with integers
            return PrimalTangentPairs(np.floor(self.primal), np.nan * self.tangent)    # ... floor, undefined
        else:                                                                          # Floor with remainder
            return PrimalTangentPairs(np.floor(self.primal), 0 * self.tangent)         # ... floor, 0

    def __ceil__(self):
        # Ceiling
        if self.primal % 1 == 0:                                                       # Ceil with integers
            return PrimalTangentPairs(np.ceil(self.primal), np.nan * self.tangent)     # ... ceil, undefined
        else:                                                                          # Ceil with remainder
            return PrimalTangentPairs(np.ceil(self.primal), 0 * self.tangent)          # ... ceil, 0

    # NumPy logarithm and exponent operators

    def exp(self):
        # Exponentiate
        return PrimalTangentPairs(np.exp(self.primal),                    # Exponentiate primal
                                  np.exp(self.primal) * self.tangent)     # ... exponential rule

    def log(self):
        # Logarithm base-e
        return PrimalTangentPairs(np.log(self.primal),                    # Natural logarithm primal
                                  self.tangent / self.primal)             # ... logarithm rule

    def log2(self):
        # Logarithm base-2
        pair = self.log()                                                 # Compute natural log
        return PrimalTangentPairs(pair.primal / np.log(2),                # Apply logarithm rule for primal
                                  pair.tangent / np.log(2))               # ... and logarithm rule for tangent

    def log10(self):
        # Logarithm base-10
        pair = self.log()                                                 # Compute natural log
        return PrimalTangentPairs(pair.primal / np.log(10),               # Apply logarithm rule for primal
                                  pair.tangent / np.log(10))              # ... and logarithm rule for tangent

    def sqrt(self):
        # Square root
        return PrimalTangentPairs(np.sqrt(self.primal),                        # Compute square root of primal
                                  self.tangent / (2 * np.sqrt(self.primal)))   # ... power rule for tangent

    # NumPy trigonometry operators

    def sin(self):
        # Sine function
        return PrimalTangentPairs(np.sin(self.primal),
                                  np.cos(self.primal) * self.tangent)

    def cos(self):
        # Cosine function
        return PrimalTangentPairs(np.cos(self.primal),
                                  -np.sin(self.primal) * self.tangent)

    def tan(self):
        # Tangent function
        return PrimalTangentPairs(np.tan(self.primal),
                                  self.tangent / np.cos(self.primal)**2)

    def arcsin(self):
        # Inverse sine function
        return PrimalTangentPairs(np.arcsin(self.primal),
                                  self.tangent / np.sqrt(1 - self.primal**2))

    def arccos(self):
        # Inverse cosine function
        return PrimalTangentPairs(np.arccos(self.primal),
                                  -1 * self.tangent / np.sqrt(1 - self.primal**2))

    def arctan(self):
        # Inverse tangent function
        return PrimalTangentPairs(np.arctan(self.primal),
                                  self.tangent / (1 + self.primal**2))

    def sinh(self):
        # Hyperbolic sine function
        return PrimalTangentPairs(np.sinh(self.primal),
                                  np.cosh(self.primal) * self.tangent)

    def cosh(self):
        # Hyperbolic cosine function
        return PrimalTangentPairs(np.cosh(self.primal),
                                  np.sinh(self.primal) * self.tangent)

    def tanh(self):
        # Hyperbolic tangent function
        return PrimalTangentPairs(np.tanh(self.primal),
                                  self.tangent / (np.cosh(self.primal) ** 2))

    def arcsinh(self):
        # Inverse hyperbolic sine function
        return PrimalTangentPairs(np.arcsinh(self.primal),
                                  self.tangent / np.sqrt(self.primal**2 + 1))

    def arccosh(self):
        # Inverse hyperbolic cosine function
        return PrimalTangentPairs(np.arccosh(self.primal),
                                  self.tangent / np.sqrt(self.primal**2 - 1))

    def arctanh(self):
        # Inverse hyperbolic tangent function
        return PrimalTangentPairs(np.arctanh(self.primal),
                                  self.tangent / (1 - self.primal**2))

    # SciPy special operators

    def polygamma(self, n):
        # Polygamma function
        return PrimalTangentPairs(float(sp.special.polygamma(n, self.primal)),
                                  self.tangent * sp.special.polygamma(n+1, self.primal))

    def normal_cdf(self):
        # Cumulative Distribution Function (CDF) of the Normal Distribution
        return PrimalTangentPairs(float(norm.cdf(self.primal)),
                                  self.tangent * norm.pdf(self.primal))

    def normal_pdf(self):
        # Probability Density Function (PDF) of the Normal Distribution
        dx_pdf = -(np.exp(-self.primal**2 / 2) * self.primal) / np.sqrt(2 * np.pi)
        return PrimalTangentPairs(float(norm.pdf(self.primal)),
                                  self.tangent * dx_pdf)
