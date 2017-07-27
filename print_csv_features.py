# Script that extracts features 

import sys
import statistics as st
from fetchdata import rawdataIterator, segmentIterator
from utils import rfft_amp_phase, max_ampl_freq, ampl_freq_range
from functools import partial
import getopt

def printCsvSegmentsFreqDict(filenames, seg_fns, freq_fns):
    '''
    Prints to stdout csv (comma-separated) output of the features computed on the temporal and frequency domain
    by the two dictionaries {name:function}
    '''
    header = ''
    for fn_name in seg_fns.keys():
        header += fn_name+','
    for fn_name in freq_fns.keys():
        header += fn_name+','
    header += 'Label'
    print(header)
    dataLabels = rawdataIterator(filenames)
    segments = segmentIterator(dataLabels,verbose=False)
    for (seg, label) in segments:
        line = ''
        for fn in seg_fns.values():
            v = fn(seg)
            line += str(v)+','
        a,p = rfft_amp_phase(seg)
        a = a[1:]   #ignore the first value, that is always extremely high (freq 0)
        for fn in freq_fns.values():
            v = fn(a)
            line += str(v)+','
        line += str(label)
        print(line)

def main():
    filenames = ['../SleepEEG/rt 233_180511(1).txt','../SleepEEG/rt 233_180511(2).txt'] #['../SleepEEG/rt 239_310511(1).txt', '../SleepEEG/rt 239_310511(2).txt' ]  

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'f:h')
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
    for o, a in opts:
        if o == '-f':
            filenames = a.split(',')
        elif o == '-h':
            print('''USAGE: python print_csv_features.py [-f <filenames>] [-h] 
            -f: comma-separated file names to process (into a single file)
            -h: show this help and quit.
            ''')
            sys.exit()

    max_theta = partial(ampl_freq_range, lowcut=5, highcut=10, aggr_fn=max) #theta wave, between 5 to 10 Hz
    max_delta = partial(ampl_freq_range, lowcut=0, highcut=4, aggr_fn=max) #delta wave, between 0 to 4 Hz
    seg_features = {'SegmentLength':len,
                    'SegmentMin':min,
                    'SegmentMax':max,
                    'SegmentAvg':st.mean,
                    'SegmentStdDev':st.pstdev}
    freq_features = {'FreqAmplMin':min,
                    'FreqAmplMax':max,
                    'FreqAmplAvg':st.mean,
                    'FreqAmplStdev':st.pstdev,
                    'FreqAmplMaxFreq':max_ampl_freq,
                    'MaxTheta':max_theta,
                    'MaxDelta':max_delta}
    printCsvSegmentsFreqDict(filenames, seg_features, freq_features)

if __name__ == '__main__':
    main()