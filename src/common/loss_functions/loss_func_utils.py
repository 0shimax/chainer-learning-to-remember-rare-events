import numpy as np
import chainer
import chainer.functions as F


def loglikelihood(location_mean, sampled_location, var):
    # negative_loglikelihood = \
    #     F.gaussian_nll(sampled_location, mean=location_mean, ln_var=sigma)

    negative_loglikelihood = -0.5*(sampled_location - location_mean)/var
    negative_loglikelihood = F.sum(negative_loglikelihood, axis=1)
    return negative_loglikelihood
