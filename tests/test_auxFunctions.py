import unittest

import numpy as np

from itertools import product

import eeglib.auxFunctions as aux

class TestAuxFuncs(unittest.TestCase):
    dictToFlat = {"asd":1, "lol":[2,3], "var":{"xd":4, "XD":5}}
    listToFlat = [0, 1, 2, [3, 4], [5, [6, 7]]]
    
    simpleDict = {"a":0, "b":1}
    simpleList = [0, 1]
    
    def test_flatData_dict(self):
        trueFlatten = {'slang_asd': 1, 'slang_lol_0': 2, 'slang_lol_1': 3,
                   'slang_var_xd': 4, 'slang_var_XD': 5}
        flatten = aux.flatData(self.dictToFlat, "slang")
        self.assertEqual(flatten, trueFlatten)
    
    def test_flatData_list(self):
        trueFlatten = {'list_0': 0, 'list_1': 1, 'list_2': 2, 'list_3_0': 3,
             'list_3_1': 4, 'list_4_0': 5, 'list_4_1_0': 6, 'list_4_1_1': 7}
        flatten = aux.flatData(self.listToFlat, "list")
        self.assertEqual(flatten, trueFlatten)
    
    def test_flatData_names(self):
        #Flat List with name
        trueFlatten = {'l_0': 0, 'l_1': 1}
        flatten = aux.flatData(self.simpleList, "l")
        self.assertEqual(flatten, trueFlatten)
        #Flat List without name
        trueFlatten = {'0': 0, '1': 1}
        flatten = aux.flatData(self.simpleList, "")
        self.assertEqual(flatten, trueFlatten)
        #Flat Dict with name
        trueFlatten = {'l_a': 0, 'l_b': 1}
        flatten = aux.flatData(self.simpleDict, "l")
        self.assertEqual(flatten, trueFlatten)
        #Flat Dict without name
        trueFlatten = {'a': 0, 'b': 1}
        flatten = aux.flatData(self.simpleDict, "")
        self.assertEqual(flatten, trueFlatten)
    
    def test_flatData_separators(self):
        #Flat List with tabs separator
        trueFlatten = {'l\t0': 0, 'l\t1': 1}
        flatten = aux.flatData(self.simpleList, "l", separator="\t")
        self.assertEqual(flatten, trueFlatten)
        #Flat List with empty separator
        trueFlatten = {'l0': 0, 'l1': 1}
        flatten = aux.flatData(self.simpleList, "l", separator="")
        self.assertEqual(flatten, trueFlatten)
        #Flat Dict with tabs separator
        trueFlatten = {'l\ta': 0, 'l\tb': 1}
        flatten = aux.flatData(self.simpleDict, "l", separator="\t")
        self.assertEqual(flatten, trueFlatten)
        #Flat Dict with empty separator
        trueFlatten = {'la': 0, 'lb': 1}
        flatten = aux.flatData(self.simpleDict, "l", separator="")
        self.assertEqual(flatten, trueFlatten)
        
    def test_listType_noList(self):
        noList = "asd"
        with self.assertRaises(ValueError):
            aux.listType(noList)
    
    def test_listType_noLength(self):
        testList = []
        v = aux.listType(testList)
        self.assertEqual(v, None)
    
    def test_listType_diffTypes(self):
        v = aux.listType([1, "2", 3.0])
        self.assertEqual(v, None)
        v = aux.listType([*range(10),"10", *range(11,21)])
        self.assertEqual(v, None)
    
    def test_listType_sameTypes(self):
        v = aux.listType([*range(10)])
        self.assertEqual(v, int)
        v = aux.listType(["a", 'b', "1", '2', """"long text"""])
        self.assertEqual(v, str)
        

if __name__ == "__main__":
    unittest.main()
 
