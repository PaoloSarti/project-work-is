import csv
import sys
import getopt
import numpy as np
from matplotlib import pyplot as plt
import statistics as st
import random
from sklearn.model_selection import train_test_split
from utils import aggregate, normalize, to4Labels
from collections import deque
import json
from os import path

def rawdata(filenames, n=-1):
    data = []
    labels = []
    i = 0
    for filename in filenames:
        with open(filename) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                data.append(float(row['1 EEG']))
                labels.append(int(float(row['10 Hypnogm'])))
                i += 1
                if i == n:
                    break
    return data, labels

def rawdataIterator(filenames, n=-1):
    """
    Yields couples (eeg, label) from the specified files.
    Optionally, a maximum number of lines can be specified,
    defaults to -1 (all the lines).
    """
    i = 0
    for filename in filenames:
        with open(filename) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                eeg = float(row['1 EEG'])
                hypno = int(float(row['10 Hypnogm']))
                yield (eeg, hypno)
                i += 1
                #print(str(eeg)+' '+str(hypno)+' '+str(i))
                if i == n:
                    f.close()
                    raise StopIteration()

def load_segment_statistics(filenames):
    data = []
    labels = []
    for filename in filenames:
        with open(filename) as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                #print(str(row))
                instance = []
                instance.append(int(row['SegmentLength']))
                instance.append(float(row['SegmentMin']))
                instance.append(float(row['SegmentMax']))
                instance.append(float(row['SegmentAvg']))
                instance.append(float(row['SegmentStdDev']))
                data.append(instance)
                labels.append(int(row['Label']))
    return data, labels

def reduceResolution(dataLabels, n):
    i = 0
    toAggregate = []
    labels = []
    for (data, label) in dataLabels:
        if i < n:
            toAggregate.append(data)
            labels.append(label)
            i += 1
        else:
            m = st.mean(toAggregate)
            l = round(st.mean(labels))
            #print('reduceRes yields: ' + str(m) + ' ' + str(l))
            yield (m,l)
            toAggregate = [data]
            labels = [label]
            i = 1
    if len(toAggregate) > 0:
        m = st.mean(toAggregate)
        l = round(st.mean(labels))
        yield (m,l)

def segmentIterator(dataLabels, n = -1, cut_until_change=True, aggr=1, transitions=False, verbose=True):
    '''
    iterates on an iterator of couples data-label
    returns a new iterator of tuples (segment-label)
    a limit in size of the segments can be specified
    '''
    prevlabel = -1
    first = True
    seg = []
    i = 0
    j = 0
    beforeLabel = -1
    if transitions:
        before = deque(maxlen=2*n)
    for (eeg, label) in dataLabels:
        if first:
            prevlabel = label
            beforeLabel = label
        elif transitions:
            before.append(eeg)
        first = False
        if label == prevlabel and (i < n or n == -1):
            seg.append(eeg)
            i += 1
        elif cut_until_change and label == prevlabel and n != -1:
            i += 1
        else:
            if transitions:
                beforelist = list(before)[:n]
            if aggr > 1:
                seg = aggregate(seg,aggr)
                if transitions:
                    beforelist = aggregate(beforelist,aggr)
            if transitions:
                seg = beforelist + seg
                yieldlabel = to4Labels(beforeLabel, prevlabel)
            else:
                yieldlabel = prevlabel
            yield (seg, yieldlabel)
            j += 1
            if verbose:
                print('Yielded Segment n '+str(j)+' of length ' + str(len(seg)) + ' with label '+str(yieldlabel))
            seg = [eeg]
            beforeLabel, prevlabel = prevlabel, label
            i = 1
    if len(seg) != 0 and not cut_until_change and n == -1 and not transitions:
        if verbose:
            print('Yielded Segment n '+str(j)+' of length ' + str(len(seg)) + ' with label '+str(label))
        if aggr > 1:
                seg = aggregate(seg,aggr)
        yield (seg,prevlabel)

def dataLabelsArrays(dataLabels):
    data = []
    labels = []
    for (d, l) in dataLabels:
        data.append(d)
        labels.append(l)
    return data, labels

def dataLabelsArraysShuffled(dataLabels):
    dataLabelsArray = [dl for dl in dataLabels]
    random.shuffle(dataLabelsArray)
    return dataLabelsArrays(dataLabelsArray)

def normalizeSegments(segments, mi=-1, ma=-1):
    nsegs = []
    for seg in segments:
        if len(seg) > 0:
            nsegs.append(normalize(seg, mi, ma))
    return nsegs

def maxMinSegments(segments):
    ma = float('-inf')
    mi = float('inf')
    for seg in segments:
        curMax = max(seg)
        curMin = min(seg)
        ma = curMax if curMax > ma else ma
        mi = curMin if curMin < mi else mi
    return ma,mi

def holdoutTrainTest(data, labels, perc_train):
    l = len(data)
    split_index = l * perc_train
    trainData = data[:split_index]
    trainLabels = labels[:split_index]
    testData = data[split_index:]
    testLabels = labels[split_index:]
    #Normalize according to the training set
    ma, mi = maxMinSegments(trainData)
    trainData = normalizeSegments(trainData, ma, mi)
    testData = normalizeSegments(testData, ma, mi)
    return (trainData, trainLabels),(testData,testLabels)

def partitionTrainValidTest(data, labels, perc_train=1/3, perc_valid=1/3):
    l = len(data)
    l1 = int(l*perc_train)
    l2 = l1 + int(l*perc_valid)
    trainData = data[:l1]
    trainLabels = labels[:l1]
    validateData = data[l1:l2]
    validateLabels = labels[l1:l2]
    testData = data[l2:]
    testLabels = labels[l2:]
    #Normalize according to the training set
    ma, mi = maxMinSegments(trainData)
    trainData = normalizeSegments(trainData, ma, mi)
    validateData = normalizeSegments(validateData, ma, mi)
    testData = normalizeSegments(testData, ma, mi)
    return (trainData,trainLabels), (validateData,validateLabels), (testData, testLabels)

def stratifiedTrainValidTest(data, labels, perc_train=0.5, perc_valid=0.2):
    trainValidData, testData, trainValidLabels, testLabels = train_test_split(data, labels, train_size=perc_train+perc_valid, stratify=labels)
    trainData, validateData, trainLabels, validateLabels = train_test_split(trainValidData, trainValidLabels, train_size=perc_train/(perc_train+perc_valid), stratify=trainValidLabels)
    print('data length: '+str(len(data))+' '+str(len(labels)))
    print('train lengths: '+str(len(trainData))+' '+str(len(trainLabels)))
    print('validate lengths: '+str(len(validateData))+' '+str(len(validateLabels)))
    print('test length: '+str(len(testData))+' '+str(len(testLabels)))
    return (trainData,trainLabels), (validateData, validateLabels), (testData, testLabels)

def load_segment_statistics_train_test(filename, perc_train=0.7):
    data, labels = load_segment_statistics(filename)
    return stratifiedTrainTest(data, labels, perc_train)

def stratifiedTrainTest(data,labels,perc_train=0.7):
    trainData, testData, trainLabels, testLabels = train_test_split(data, labels, train_size=perc_train, stratify=labels)
    return (trainData,trainLabels), (testData, testLabels)

def loadSegmentsLabels(filenames, res, n=-1, shuffle=True):
    dataLabels = rawdataIterator(filenames, n)
    if res != 1:
        dataLabels = reduceResolution(dataLabels, res)
    segments = segmentIterator(dataLabels)
    if shuffle:
        return dataLabelsArraysShuffled(segments)
    else:
        return dataLabelsArrays(segments)

def loadTrainTest(filenames, res, n=-1):
    data, labels = loadSegmentsLabels(filenames, res, n)
    return holdoutTrainTest(data, labels, 2/3)

def loadTrainValidationTest(filenames, res=1, n=-1):
    data, labels = loadSegmentsLabels(filenames, res, n)
    return partitionTrainValidTest(data, labels)

def loadStratifiedDataset(filenames, aggr=1, n=-1, seconds=5, sampling_period=0.002, validation=False, transitions=False, verbose=True):
    #caching the intermediate result
    dumpedfilename = 'cache' + '_a'+str(aggr)+'_n'+str(n)+'_s'+str(seconds)
    dumpedfilename = dumpedfilename + '_t.json' if transitions else dumpedfilename + '.json'
    data, labels = [], []
    if path.isfile(dumpedfilename):
        with open(dumpedfilename) as df:
            data, labels = json.load(df)
    else:
        dataLabels = rawdataIterator(filenames, n)
        if seconds != -1:
            segments = segmentIterator(dataLabels, n=int(seconds/sampling_period),aggr=aggr, transitions=transitions, verbose=verbose)
        else:
            segments = segmentIterator(dataLabels, cut_until_change=False, n=-1,aggr=aggr, transitions=transitions, verbose=verbose)
        dataLabArrays = dataLabelsArrays(segments)
        with open(dumpedfilename, 'w') as df:
            json.dump(dataLabArrays,df)
        data, labels = dataLabArrays
    if validation:
        (trainData,trainLabels), (validateData, validateLabels), (testData, testLabels) = stratifiedTrainValidTest(data, labels)
        ma, mi = maxMinSegments(trainData)
        trainData = normalizeSegments(trainData, ma, mi)
        validateData = normalizeSegments(validateData, ma, mi)
        testData = normalizeSegments(testData, ma, mi)
        return (trainData,trainLabels), (validateData, validateLabels), (testData, testLabels)
    else:
        (trainData,trainLabels), (testData, testLabels) = stratifiedTrainTest(data, labels)
        ma, mi = maxMinSegments(trainData)
        trainData = normalizeSegments(trainData, ma, mi)
        testData = normalizeSegments(testData, ma, mi)
        return (trainData,trainLabels), (testData, testLabels)

def segment(data, labels):
    lists = []
    listsLabels = []
    l = len(data)
    prevlabel = -1
    i = 0
    j = 0
    while i < l:
        prevlabel = labels[i]
        lists.append([])
        listsLabels.append(labels[i])
        while i < l and labels[i] == prevlabel:
            lists[j].append(data[i])
            i += 1
        j += 1
    return lists, listsLabels

def visualize(data, step):
    xs = np.arange(0,len(data)*step,step)
    plt.plot(xs, data)
    plt.xlabel('time (s)')
    plt.ylabel('Voltage mV')
    plt.grid(True)
    plt.show()

def saveSegmentsFigs(segmentedData, labels, step, basefilename):
    for i in range(len(segmentedData)):
        xs = np.arange(0,len(segmentedData[i])*step,step)
        plt.plot(xs, segmentedData[i])
        plt.xlabel('time (s)')
        plt.ylabel('Voltage mV')
        plt.grid(True)
        plt.savefig(basefilename + '_' + str(i) + '_' + str(labels[i]) + '.png')
        plt.clf()


def segmentWithLabelAndLength(data, labels, n):
    lists = []
    listsLabels = []
    l = len(data)
    prevlabel = -1
    i = 0
    j = 0
    while i < l:
        prevlabel = labels[i]
        lists.append([])
        listsLabels.append(labels[i])
        k = 0
        while i < l and labels[i] == prevlabel and k < n:
            lists[j].append(data[i])
            i += 1
            k += 1
        j += 1
    return lists, listsLabels

#--------------------------------Tests-------------------------------

def printCsvSegments(filenames, n):
    data, labels = rawdata(filenames, n)
    dl, ll = segment(data, labels)
    length = len(ll)
    print('SegmentLength,Label')
    for i in range(length):
        print(str(len(dl[i])) + ',' + str(ll[i]))

def printCsvSegmentsIterator(filenames, n):
    dataLabels = rawdataIterator(filenames,n)
    segments = segmentIterator(dataLabels)
    print('SegmentLength,Label')
    for (seg,label) in segments:
        print(str(len(seg)) + ',' + str(label))

def printCsvSegmentsReduceRes(filenames, n, r):
    dataLabels = rawdataIterator(filenames,n)
    if r != 1:
        dataLabels = reduceResolution(dataLabels, r)
    segments = segmentIterator(dataLabels)
    print('SegmentLength,SegmentMin,SegmentMax,SegmentAvg,SegmentStdDev,Label')
    for (seg,label) in segments:
        l = len(seg)
        mi = min(seg)
        ma = max(seg)
        avg = st.mean(seg)
        stdev = st.pstdev(seg,avg)
        print(str(l) + ',' + str(mi) + ',' + str(ma) + ',' + str(avg) + ',' + str(stdev) +',' + str(label))

def visualizeResReduction(filenames, n=-1, res=1):
    dataLabels = rawdataIterator(filenames, n)
    dataLabels = reduceResolution(dataLabels, res)
    data, labels = dataLabelsArrays(dataLabels)
    segn = normalize(data)
    visualize(segn, 0.002*res)

def visualizeSeconds(filenames, n=-1, res=1, seconds=-1):
    dataLabels = rawdataIterator(filenames, n)
    if res != 1:
        dataLabels = reduceResolution(dataLabels, res)
    segmentsLabels = segmentIterator(dataLabels,int(seconds/(0.002*res)))
    for (seg, label) in segmentsLabels:
        print('Label: ' + str(label))
        segn = normalize(seg)
        visualize(segn,0.002*res)

def main():
    #filename = '../SleepEEG/rt 233_180511(1).txt'
    #filename1 = '../SleepEEG/rt 233_180511(2).txt'
    filenames = ['../SleepEEG/rt 239_310511(1).txt','../SleepEEG/rt 239_310511(2).txt']
    n = -1 #all data
    r = 1 #no reduction #250
    seconds = 5
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'f:n:r:s:h')
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
    for o, a in opts:
        if o == '-f':
            filenames = a.split(',')
        elif o == '-n':
            n = int(a)
        elif o == '-r':
            r = int(a)
        elif o == '-s':
            seconds = int(a)
        elif o == '-h':
            print('USAGE: python fetchdata.py [-n <lines> | -f <filename> | -r <reduceRes> | -h]')
            sys.exit()
        else:
            sys.exit(3)

    print(str(filenames))
    print(str(n))
    print(str(r))
    print(str(seconds))
    #printCsvSegments(filename,n)
    #printCsvSegmentsIterator([filename,filename1],n)
    #printCsvSegmentsReduceRes([filename,filename1],n, r)
    #visualizeResReduction(filenames,n,r)
    visualizeSeconds(filenames,n,r,seconds)
    
if __name__ == '__main__':
    main()