import torch.multiprocessing as mp


def kwargs_pool(func, kwargs):
    '''Helper function for use of kwargs with starmap.'''
    return func(**kwargs)


def ptmultiproc(func, iterator_keys, n_procs=None, **kwargs):
    '''
    A general function to use multiprocessing in PyTorch.

    Args:
        func (function): A function that will be performed in parallel.
        iterator_keys (str): A list of keys in kwargs whose values are 
            the iterables to divide up during multiprocessing. All 
            iterables in kwargs indicated by iterator_keys should have 
            the same number of elements. kwargs not indicated by 
            iterator_keys will not be divided up during 
            multiprocessing, and instead will be replicated for every
            function call.
        n_procs (int): The number of processes to use. Be careful with 
            this.
        kwargs: keyword arguments to func. The items specified by 
            iterator_keys should be iterables.

    Returns:
        result (list): A list of n elements, where each element is the
            output of func and n is the length of the iterables 
            specified by iterator_keys.
    '''

    # get length of iterables
    n_inputs = len(kwargs[iterator_keys[0]])
    if not all([len(kwargs[key]) == n_inputs for key in iterator_keys]):
        raise ValueError
    
    # create the arg list for starmap
    args = []
    for input_num in range(n_inputs):
        kwargs_copy = kwargs.copy()

        for key in iterator_keys:
            kwargs_copy[key] = kwargs_copy[key][input_num]
            
        args.append((func, kwargs_copy))

    with mp.Pool(n_procs) as pool:
        result = pool.starmap(kwargs_pool, args)

    return result