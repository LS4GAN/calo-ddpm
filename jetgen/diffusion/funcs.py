
def match_shape(param, target):
    # param     : (N, )
    # target    : (N, ...)

    if param.ndim != 1:
        return param

    param = param.view((param.shape[0], ) + (1,) * (target.dim() - 1))
    return param
    #return param.expand_as(target)

