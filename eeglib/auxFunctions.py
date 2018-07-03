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
