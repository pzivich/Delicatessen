import numpy as np


def auto_differentiation(xk, f):
    """Forward mode automatic differentiation. Automatic differentiation offers a way to compute the derivative exactly,
    rather than numerically approximated (unlike the central difference method). Automatic differentiation iteratively
    applies the chain rule into order to evaluate the derivative.

    Note
    ----
    This functionality is only intended for use behind the scenes in ``delicatessen``. I wrote this functionality to
    avoid additional dependencies.

    Parameters
    ----------
    xk: ndarray, list, shape (n, )
        Point(s) or coordinate vector to evaluate the gradient at.
    f: callable
        Function of which to estimate the gradient of.

    Returns
    -------
    ndarray

    Examples
    --------

    References
    ----------

    """
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
        eval = function(x)                 # Short-name for function
        eval_no_bool_end = []              # Empty list for storage
        for e in eval:                     # For each evaluation in the function
            eval_no_bool_end.append(e+0)   # ... adding zero in a loop
        return eval_no_bool_end            # Return the evaluation with the final addition

    # Meta-information about function and inputs
    xshape = len(xk)                                                   # The number of inputs into the function

    # Set up Dual objects for evaluating gradient of function
    pairs_for_gradient = []                                            # Storage for the pairs to provide function
    for i in range(xshape):                                            # For each of the inputs
        partial = np.zeros_like(xk)                                    # ... generate array of that size of all zeroes
        partial[i] = 1                                                 # ... replace 0 with 1 for current index
        pairs_for_gradient.append(PrimalTangentPairs(xk[i], partial))  # ... then store as a primal,tangent pair
    x_to_eval = np.asarray(pairs_for_gradient)                         # Convert from list to NumPy array

    # Evaluating the function for the primal,tangent pairs
    evaluated_pair = f_deny_bool(function=f,                           # Evaluate the primal,tangent pair with function
                                 x=x_to_eval)                          # ... at the given values

    # Processing function output into gradient value or matrix
    evaluated_gradient = []                                            # List storage for the computed gradient
    for pair in evaluated_pair:                                        # For each evaluated primal,tangent pair
        if isinstance(pair, PrimalTangentPairs):                       # ... if that pair is the key type
            evaluated_gradient.append(pair.tangent)                    # ... then give back the derivative
        else:                                                          # ... otherwise row has no xk operations
            evaluated_gradient.append([0 for j in range(xshape)])      # ... so derivative is always zero

    # Return evaluated gradient as a NumPy array
    if len(evaluated_gradient) == 1:                                   # If only consists of 1 item
        return np.asarray(evaluated_gradient[0])                       # ... return that item as NumPy array
    else:                                                              # Otherwise
        return np.asarray(evaluated_gradient)                          # ... return is as an array


class PrimalTangentPairs:
    """Unique class object for automatic differentiation. This process divides the inputs into 'primal' and 'tangent'
    pairs. Operations on the pairs are then recurvsively called.


    Parameters
    ----------
    primal :
        B
    tangent :
        A
    """
    def __init__(self, primal, tangent):
        # Processing of the inputs
        if isinstance(primal, PrimalTangentPairs):    # If given a PrimalTangentPair input
            self.primal = primal.primal               # ... extract the primal element from input
        else:                                         # Else
            self.primal = primal                      # ... directly save as new primal
        self.tangent = tangent                        # Store the tangent

    # Basic operators

    def __str__(self):
        # Conversion to string just to in case it gets called somehow
        return f"PrimalTangentPairs({self.primal}, {self.tangent})"

    def __bool__(self):
        # To get np.where working properly, I need to have the internal boolean function for this class return
        #   only the primal part. This is a bool object type, which is what is expected and has np.where operate as
        #   expected. This only seems to be called for np.where and not the other operators (they work directly).
        return self.primal

    # Equality operators

    def __le__(self, other):
        # Less than or equal to operator
        if isinstance(other, PrimalTangentPairs):                  #
            comparison = (self.primal <= other.primal)             #
        else:                                                      #
            comparison = (self.primal <= other)                    #
        return PrimalTangentPairs(comparison, 0)                   #

    def __lt__(self, other):
        # Less than operator
        if isinstance(other, PrimalTangentPairs):
            comparison = (self.primal < other.primal)
        else:
            comparison = (self.primal < other)
        return PrimalTangentPairs(comparison, 0)

    def __ge__(self, other):
        # Greater than or equal to operator
        if isinstance(other, PrimalTangentPairs):
            comparison = (self.primal >= other.primal)
        else:
            comparison = (self.primal >= other)
        return PrimalTangentPairs(comparison, 0)

    def __gt__(self, other):
        # Greater than operator
        if isinstance(other, PrimalTangentPairs):
            comparison = (self.primal > other.primal)
        else:
            comparison = (self.primal > other)
        return PrimalTangentPairs(comparison, 0)

    def __eq__(self, other):
        # Equality operator
        if isinstance(other, PrimalTangentPairs):
            comparison = (self.primal == other.primal)
        else:
            comparison = (self.primal == other)
        return PrimalTangentPairs(comparison, 0)

    def __ne__(self, other):
        # Inequality operator
        if isinstance(other, PrimalTangentPairs):
            comparison = (self.primal != other.primal)
        else:
            comparison = (self.primal != other)
        return PrimalTangentPairs(comparison, 0)

    # Elementary mathematical operations

    def __neg__(self):
        # Negation operator
        return PrimalTangentPairs(-self.primal,
                                  -self.tangent)

    def __add__(self, other):
        # Addition
        if isinstance(other, PrimalTangentPairs):
            return PrimalTangentPairs(self.primal + other.primal,
                                      self.tangent + other.tangent)
        else:
            return PrimalTangentPairs(self.primal + other,
                                      self.tangent)

    def __radd__(self, other):
        # Addition, reversed
        return self.__add__(other)

    def __sub__(self, other):
        # Subtraction
        if isinstance(other, PrimalTangentPairs):
            return PrimalTangentPairs(self.primal - other.primal,
                                      self.tangent - other.tangent)
        else:
            return PrimalTangentPairs(self.primal - other,
                                      self.tangent)

    def __rsub__(self, other):
        # Subtraction, reversed
        return PrimalTangentPairs(other - self.primal,
                                  -1 * self.tangent)

    def __mul__(self, other):
        # Multiplication
        if isinstance(other, PrimalTangentPairs):
            return PrimalTangentPairs(self.primal * other.primal,
                                      self.tangent * other.primal + self.primal * other.tangent)
        else:
            return PrimalTangentPairs(self.primal * other,
                                      self.tangent * other)

    def __rmul__(self, other):
        # Multiplication, reversed
        return PrimalTangentPairs(self.primal * other,
                                  self.tangent * other)

    def __truediv__(self, other):
        # Division
        if isinstance(other, PrimalTangentPairs):
            return PrimalTangentPairs(self.primal / other.primal,
                                      (self.tangent * other.primal - self.primal * other.tangent) / (other.primal ** 2))
        else:
            return PrimalTangentPairs(self.primal / other,
                                      self.tangent / other)

    def __rtruediv__(self, other):
        # Division, reversed
        return PrimalTangentPairs(other / self.primal,
                                  (0 * self.primal - self.tangent * other) / (self.primal ** 2))

    def __pow__(self, other):
        # Power
        if isinstance(other, PrimalTangentPairs):
            fgx = other.primal * self.primal ** (other.primal - 1) * self.tangent
            gx = self.primal ** other.primal * np.log(self.primal) * other.tangent
            return PrimalTangentPairs(self.primal ** other.primal,
                                      fgx + gx)
        else:
            return PrimalTangentPairs(self.primal ** other,
                                      other * self.primal ** (other - 1) * self.tangent)

    def __rpow__(self, other):
        # Power, reversed
        return PrimalTangentPairs(other ** self.primal,
                                  other ** self.primal * np.log(other))

    # Operators that sometimes have undefined derivatives

    def __abs__(self):
        # Absolute value
        if self.primal > 0:
            return PrimalTangentPairs(1 * self.primal, self.tangent)
        elif self.primal < 0:
            return PrimalTangentPairs(-1 * self.primal, -1 * self.tangent)
        else:
            return PrimalTangentPairs(0 * self.primal, np.nan * self.tangent)

    def __floor__(self):
        # Floor
        if self.primal % 1 == 0:
            return PrimalTangentPairs(np.floor(self.primal), np.nan * self.tangent)
        else:
            return PrimalTangentPairs(np.floor(self.primal), 0 * self.tangent)

    def __ceil__(self):
        # Ceiling
        if self.primal % 1 == 0:
            return PrimalTangentPairs(np.ceil(self.primal), np.nan * self.tangent)
        else:
            return PrimalTangentPairs(np.ceil(self.primal), 0 * self.tangent)

    # NumPy logarithm and exponent operators

    def exp(self):
        # Exponentiate
        return PrimalTangentPairs(np.exp(self.primal),
                                  np.exp(self.primal) * self.tangent)

    def log(self):
        # Logarithm
        return PrimalTangentPairs(np.log(self.primal),
                                  self.tangent / self.primal)

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
                                  self.tangent / np.cos(self.primal))

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
                                  (1 / (np.cosh(self.primal) ** 2)) * self.tangent)

    # TODO add arc trig functions

