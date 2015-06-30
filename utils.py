__author__ = 'michaelpearmain'

def logloss(p, y):
    ''' FUNCTION: Bounded logloss
        INPUT:
            p: our prediction
            y: real answer
        OUTPUT:
            logarithmic loss of p given y
    '''

    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)

def feature_hash(v, n=32):
    """ Get n-bit hash value

        INPUT:
            v: value to be hashed
            n: number of bits (n<=32)

        OUTPUT:
            hash value no longer than n
     """
    return abs(mmh3.hash(v)) >> (32-n)

def norm_pdf(x):
    """
    Computes the probability density functions of the value in x using the normal distribution with mean mu=0
     and standard deviation sigma=1.
    :param x: A positive value to compute the pdf
    :return: The pdf of the value x
    """
    return exp(-x * x /2.) / sqrt(2.*pi)


def norm_cdf(x):
    """
    Computes the Normal cumulative distribution function of the value in x using the normal distribution with mean mu=0
     and standard deviation sigma=1.
    :param x: A value for which to compute the cdf
    :return: The cdf of x
    """
    return 1. - 0.5 * erfc(x/(2**0.5))


def gaussian_corrections(t):
    """
    Returns the additive and multiplicative corrections for the mean and variance of a trunctated Gaussian random
    variable.
    In Trueskill/AdPredictor papers, denoted
    - V(t)
    - W(t) = V(t) * (V(t) + t)
    Returns (v(t), w(t))
    """
    # Clipping avoids numerical issues from ~0/~0.
    t = max(-5.0, min(5.0, t))
    v = norm_pdf(t) / norm_cdf(t)
    w = v * (v + t)
    return (v, w)


def kl_divergence(p, q):
    """
    Computes the Kullback-Liebler divergence between two Bernoulli random variables with probability p and q.
    Algebraically, KL(p || q)
    Specifically, the Kullback-Leibler divergence of q from p, is a measure of the information lost when q is used to
    approximate p

    :param p: The probability of a Bernoulli random variable with probability p
    :param q: The probability of a Bernoulli random variable with probability q
    :return: The divergence metric between the two variables.
    """
    return p * log(p / q) + (1.0 - p) * log((1.0 - p) / (1.0 - q))


def label_to_float(label):
    """
    Helper function to cast the label of interest (click or no click / conversion or no conversion) to a float
    :param label: Value of label 1 => Postive respose to bool, negative
    :return:
    """
    assert type(label) == bool
    return 1.0 if label else -1.0


def ascii_encode(x):
    """
    Change encoding to a standard format.

    :param x: Character vector to be encoded.
    :return: ascii encoding of character vector x.
    """
    if type(x) == unicode:
        return x.encode("ascii")
    elif type(x) == list:
        return map(ascii_encode, x)
    elif type(x) == dict:
        return ascii_encode_dict(x)
    else:
        return x

def ascii_encode_dict(data):
    """
    Helper function to encode all data to ascii by calling the accii_encode function.
    :param data: The row of data to be encoded to ascii
    :return: ascii encoded data row
    """
    return dict(map(ascii_encode, pair) for pair in data.items())
