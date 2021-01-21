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
    if not isinstance(l, list):
        raise ValueError("l is not a list.")

    if len(l) == 0:
        return None

    t = type(l[0])
    for e in l:
        if not isinstance(e, t):
            return None

    return t


def flatData(data, name, separator = "_"):
    """
    Transforms the parameter value into a flatten dictionary, using the
    parameter name as a prefix for the dictionary keys.

    """
    res={}

    auxName=name
    if name != "":
        auxName = auxName + separator

    if isinstance(data, dict):
        for key, val in data.items():
            res.update(flatData(val, auxName+str(key), separator))
    elif hasattr(data, "__iter__"):
        for i, val in enumerate(data):
            res.update(flatData(val, auxName+"%d"%i, separator))
    else:
        res = {name:data}

    return res
