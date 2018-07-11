"""
This module includes auxiliar functions that are intended to be used only by 
the library.
"""
def listType(l):
    """
    If the type of every element of the list l is the same, this function 
    returns that type, else it returns None. If l is not a list, this function
    will raise a ValueError.
    """
    if type(l) is not list:
        raise ValueError("l is not a list.")
    
    if len(l) == 0:
        return None
    
    t = type(l[0])
    for e in l:
        if type(e) != t:
            return None
    
    return t


def flat(value, name):
    """
    Transforms the parameter value into a flatten dictionary, using the
    parameter name as a prefix for the dictionary keys. 
         
    """
    res={}
    if type(value) is dict:
        for key, val in value.items():
            res.update(flat(val, name + "_"+str(key)))
    elif hasattr(value, "__iter__"):   
        for i, val in enumerate(value):
            res.update(flat(val, name+"_%d"%i))
    else:
        res = {name:value}
    
    return res