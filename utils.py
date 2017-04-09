import statistics as st
import numpy as np
from collections import deque
from os import path
import json
import math

def aggregate(array, n=1, fn=st.mean):
    i = 0
    toAggregate = []
    aggregated = []
    for e in array:
        if i < n:
            toAggregate.append(e)
            i += 1
        else:
            m = fn(toAggregate)
            aggregated.append(m)
            toAggregate = [e]
            i = 1
    if len(toAggregate) > 0:
        m = fn(toAggregate)
        aggregated.append(m)
    return aggregated

def normalize(l, mi=-1, ma=-1):
    if mi == -1:
        mi = min(l)
    if ma == -1:
        ma = max(l)
    n = []
    for d in l:
        if ma - mi != 0:
            n.append((d - mi)/(ma - mi))
    return n

def normalizeColumns(data):
    npdata = np.array(data)
    cols = [npdata[:,j] for j in range(npdata.shape[1])]
    mins = list(map(min, cols))
    maxs = list(map(max, cols))
    norm = np.zeros(npdata.shape)
    for i in range(npdata.shape[0]):
        for j in range(npdata.shape[1]):
            norm[i,j] = (npdata[i,j]-mins[j])/(maxs[j]-mins[j])
    return norm

def to4Labels(prevLabel, curLabel):
    dic = {'10':0,   #nrem-awake
           '01':1,   #awake-nrem
           '12':2,   #nrem-rem
           '20':3,   #rem-awake
           '00':0,   #awake-awake (not found, but useful at the start)
           '11':1}   #nrem-nrem (not found, but useful at the start)
    return dic[str(prevLabel)+str(curLabel)]

def labels(n_classes=3):
    return ['awake','nrem','rem'] if n_classes == 3 else ['nrem-awake','awake-nrem','nrem-rem', 'rem-awake']

def json_file_cached(fn):
    '''
    Returns a function that caches the result in a json file,
    so, if it is called multiple times with the same arguments,
    the first time calculates the result and saves it in a json file.
    The following times returns the cached result.
    Can be used as an annotation
    '''
    def cached_fun(*args,**kwargs):
        filename = 'cache'+'_'+fn.__name__
        for a in args:
            filename += '_'+str(a)
        for key,arg in kwargs.items():
            filename += '_'+key + str(arg)
        filename += '.json'
        if path.isfile(filename):
            with open(filename) as fd:
                    return json.load(fd)
        else:
            result = fn(*args, **kwargs)
            with open(filename, 'w') as fd:
                json.dump(result, fd)
            return result
    return cached_fun

def print_parameters(indent, **kwargs):
    for key,arg in kwargs.items():
        print(indent+key+': '+str(arg))
    print()

#Thanks to Zach Guo
def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels]+[5]) # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=' ')
    for label in labels: 
        print("%{0}s".format(columnwidth) % label,end=' ')
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=' ')
        for j in range(len(labels)): 
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=' ')
        print()

def items_count(l):
    d = dict()
    for i in l:
        if i in d.keys():
            d[i] += 1
        else:
            d[i] = 1
    return d

def normalize_counts(d):
    l = sum(i for i in d.values())
    nd = {}
    for k,v in d.items():
        nd[k] = v/l
    return nd

def invert_counts(d):
    nd = {}
    for k,d in d.items():
        nd[k] = 1.0/d
    return nd

def inverted_counts(l):
    return invert_counts(items_count(l))

def softmax_dict(d):
    nd = {}
    den = sum(math.exp(z) for z in d.values())
    for k,v in d.items():
        nd[k] = math.exp(v)/den
    return nd

def class_weight_count(labels):
    counts = items_count(labels)
    #print('Label counts: '+str(counts))
    nc = normalize_counts(counts)
    #print('Normalized counts: '+str(nc))
    ic = invert_counts(counts)
    #print('Inverted counts: '+ str(ic))
    inc = invert_counts(nc)
    #print('Inverted normalized counts: '+ str(inc))
    return normalize_counts(inc)