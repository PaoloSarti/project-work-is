# Module that contains utility functions

import statistics as st
import numpy as np
from collections import deque
from os import path
import json
import math
from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter

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
    if ma == mi:
        return [0 for i in l]
    else:
        return [(d - mi)/(ma - mi) for d in l]

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

def label_names(n_classes=3):
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

#Thanks to Zach Guo https://gist.github.com/zachguo/10296432
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

def reshape(dataset, seq_len, n_features):
    l = dataset.shape[0]
    return dataset.reshape((l,seq_len,n_features))

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

def class_weights_max(labels):
    counts = items_count(labels)
    max_count = max(counts.values())
    weights = dict()
    for key,value in counts.items():
        weights[key] = max_count / counts[key]
    return weights

def class_weights_complement(labels):
    counts = items_count(labels)
    sum_counts = sum(counts.values())
    weights = dict()
    for key, value in counts.items():
        others_counts = sum(v for k,v in counts.items() if k != key)
        weights[key] = others_counts / sum_counts
    return weights

def rfft_amp_phase(signal):
    ft = np.fft.rfft(signal)
    ap = np.array([[abs(c),np.angle(c)] for c in ft]).T
    return (ap[0], ap[1])

def plot_amp_phase(amp, phase):
    plt.figure(1)
    a = plt.subplot(211)
    #a.set_xscale('log')
    a.set_xlabel('frequncy [Hz]')
    a.set_ylabel('|amplitude|')
    a.plot(amp)
    b = plt.subplot(212)
    b.set_xlabel('frequency [Hz]')
    b.set_ylabel('Phase')
    plt.plot(phase)
    plt.show()

def join_args(sep, *args):
    return sep.join(str(i) for i in args)

# From Butterworth bandpass recipe
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def downsample(data, times):
    ds = []
    for i in range(0, len(data), times):
        ds.append(data[i])
    return ds

def max_ampl_freq(amp, ignore_first = False):
    m = -1
    j = -1
    start = 1 if ignore_first else 0
    for i in range(start, len(amp)):
        if amp[i] > m:
            m = amp[i]
            j = i
    return j

def ampl_freq_range(amp, lowcut, highcut, aggr_fn = np.mean):
    '''
    From the amplitude frequency spectrum, select a range of frequencies between highcut and lowcut, and aggregate them (default = mean)
    '''
    cut_amp = amp[int(lowcut):int(highcut)]
    return aggr_fn(cut_amp)