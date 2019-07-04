import os
import math
import numpy as np

from ekdist import ekscn

class SingleChannelRecord(object):
    """
    A wrapper over a list of time intervals 
    from idealised single channel record.
    """
    def __init__(self, verbose=False):
        self.verbose = verbose
        if self.verbose:
            print('A new record initialised.')
        self.origin = None
        self.is_loaded = False
        self.record_type = None
        
        self.badopen=-1

    def load_intervals_from_list(self, intervals, amplitudes, flags=None):
        self.itint, self.iampl = intervals, amplitudes
        if flags is not None:
            self.iprop = flags
        else:
            self.iprop = np.array([0] * len(intervals))
        self.origin = "Intervals loaded from list"
        self.is_loaded = True
        self._tres = 0.0
        self.rtint, self.rampl, self.rprop = self.itint, self.iampl, self.iprop
        self._set_periods()
            
    def load_SCN_file(self, infiles):
        """Load shut and open intervals from SCN file."""
        #TODO: check if infiles is valid entry: single file or a list of files 
        if isinstance(infiles, str):
            if os.path.isfile(infiles):
                infile = infiles
        elif isinstance(infiles, list):
            if os.path.isfile(infiles[0]):
                infile = infiles[0]
        #TODO: enable taking several scan files and join in a single record.
        # Just a single file could be loaded at present.
        self.header = ekscn.read_header(infile, self.verbose)
        self.itint, iampl, self.iprop = ekscn.read_data(
            infile, self.header)
        self.iampl = iampl.astype(float) * self.header['calfac2']
        self.origin = "Intervals loaded from SCN file: " + infile
        self.is_loaded = True
        self._tres = 0.0
        self.rtint, self.rampl, self.rprop = self.itint, self.iampl, self.iprop
        self._set_periods()
        
    def print_all_record(self):
        for i in range(len(self.itint)):
            print (i, self.itint[i], self.iampl[i], self.iprop[i])
            
    def print_resolved_intervals(self):
        print('\n#########\nList of resolved intervals:\n')
        for i in range(len(self.rtint)):
            print (i+1, self.rtint[i]*1000, self.rampl[i], self.rprop[i])
        print('\n###################\n\n')
        
    def __repr__(self):
        """String representation of SingleChannelRecord instance."""
        if not self.is_loaded:
            str_repr = "Empty record" 
        else:
            str_repr = self.origin
            str_repr += "\nTotal number of intervals = {0:d}".format(
                len(self.itint))
            str_repr += ('\nResolution for HJC calculations = ' + 
                '{0:.1f} microseconds'.format(self._tres*1e6))
            str_repr += "\nNumber of resolved intervals = {0:d}".format(
                len(self.rtint))
            str_repr += "\nNumber of time periods = {0:d}".format(
                len(self.ptint))
            str_repr += '\n\nNumber of open periods = {0:d}'.format(len(self.opint))
            str_repr += ('\nMean and SD of open periods = {0:.9f} +/- {1:.9f} ms'.
                format(np.average(self.opint)*1000, np.std(self.opint)*1000))
            str_repr += ('\nRange of open periods from {0:.9f} ms to {1:.9f} ms'.
                format(np.min(self.opint)*1000, np.max(self.opint)*1000))
            str_repr += ('\n\nNumber of shut intervals = {0:d}'.format(len(self.shint)))
            str_repr += ('\nMean and SD of shut periods = {0:.9f} +/- {1:.9f} ms'.
                format(np.average(self.shint)*1000, np.std(self.shint)*1000))
            str_repr += ('\nRange of shut periods from {0:.9f} ms to {1:.9f} ms'.
                format(np.min(self.shint)*1000, np.max(self.shint)*1000))
        return str_repr
    
    def _set_resolution(self, tres=0.0):
        self._tres = tres
        self._impose_resolution()
        self._set_periods()
    def _get_resolution(self):
        return self._tres
    tres = property(_get_resolution, _set_resolution)
    
    def _impose_resolution(self):
        """
        Impose time resolution.
        First interval to start has to be resolvable, usable and preceded by
        an resolvable interval too. Otherwise its start will be defined by
        unresolvable interval and so will be unreliable.
        (1) A concantenated shut period starts with a good, resolvable
            shutting and ends when first good resolvable opening found.
            Length of concat shut period = sum of all durations before the
            resolved opening. Amplitude of concat shut period = 0.
        (2) A concantenated open period starts with a good, resolvable opening
            and ends when first good resolvable interval is found that
            has a different amplitude (either shut or open but different
            amplitude). Length of concat open period = sum of all concatenated
            durations. Amplitude of concatenated open period = weighted mean
            amplitude of all concat intervals.
        First interval in each concatenated group must be resolvable, but may
        be bad (in which case all group will be bad).
        """

        # Find negative intervals and set them unusable
        self.iprop[np.where(self.itint < 0.0)] = 8
        # Find first resolvable and usable interval.
        n = np.intersect1d(np.where(self.itint > self._tres),
            np.where(self.iprop < 8))[0]
        
        # Initiat lists holding resolved intervals and their amplitudes and flags
        rtint, rampl, rprops = [], [], []
        # Set variables holding current interval values
        ttemp, otemp = self.itint[n], self.iprop[n]
        if (self.iampl[n] == 0):
            atemp = 0
        elif self.record_type == 'simulated':
            atemp = self.iampl[n]
        else:
            atemp = self.iampl[n] * self.itint[n]
        isopen = True if (self.iampl[n] != 0) else False
        n += 1

        # Iterate through all remaining intervals
        while n < (len(self.itint)):
            if self.itint[n] < self._tres: # interval is unresolvable

                if (len(self.itint) == n + 1) and self.iampl[n] == 0 and isopen:
                    rtint.append(ttemp)
#                    if self.record_type == 'simulated':
#                        rampl.append(atemp)
#                    else:
#                        rampl.append(atemp / ttemp)
                    rampl.append(atemp / ttemp)
                    rprops.append(otemp)
                    isopen = False
                    ttemp = self.itint[n]
                    atemp = 0
                    otemp = 8

                else:
                    ttemp += self.itint[n]
                    if self.iprop[n] >= 8: otemp = self.iprop[n]
                    if isopen: #self.iampl[n] != 0:
                        atemp += self.iampl[n] * self.itint[n]

            else:
                if (self.iampl[n] == 0): # next interval is resolvable shutting
                    if not isopen: # previous interval was shut
                        ttemp += self.itint[n]
                        if self.iprop[n] >= 8: otemp = self.iprop[n]
                    else: # previous interval was open
                        rtint.append(ttemp)
                        if self.record_type == 'simulated':
                            rampl.append(atemp)
                        else:
                            rampl.append(atemp / ttemp)
                        if (self.badopen > 0 and rtint[-1] > self.badopen):
                            rprops.append(8)
                        else:
                            rprops.append(otemp)
                        ttemp = self.itint[n]
                        otemp = self.iprop[n]
                        isopen = False
                else: # interval is resolvable opening
                    if not isopen:
                        rtint.append(ttemp)
                        rampl.append(0)
                        rprops.append(otemp)
                        ttemp, otemp = self.itint[n], self.iprop[n]
                        if self.record_type == 'simulated':
                            atemp = self.iampl[n]
                        else:
                            atemp = self.iampl[n] * self.itint[n]
                        isopen = True
                    else: # previous was open
                        if self.record_type == 'simulated':
                            ttemp += self.itint[n]
                            if self.iprop[n] >= 8: otemp = self.iprop[n]
                        elif (math.fabs((atemp / ttemp) - self.iampl[n]) <= 1.e-5):
                            ttemp += self.itint[n]
                            atemp += self.iampl[n] * self.itint[n]
                            if self.iprop[n] >= 8: otemp = self.iprop[n]
                        else:
                            rtint.append(ttemp)
                            rampl.append(atemp / ttemp)
                            if (self.badopen > 0 and rtint[-1] > self.badopen):
                                rprops.append(8)
                            else:
                                rprops.append(otemp)
                            ttemp, otemp = self.itint[n], self.iprop[n]
                            atemp = self.iampl[n] * self.itint[n]

            n += 1
        # end of while

        # add last interval
        if isopen:
            rtint.append(-1)
        else:
            rtint.append(ttemp)
        rprops.append(8)
        if isopen:
            if self.record_type == 'simulated':
                rampl.append(atemp)
            else:
                rampl.append(atemp / ttemp)
        else:
            rampl.append(0)
        self.rtint, self.rampl, self.rprop = rtint, rampl, rprops

    def _set_periods(self):
        self.periods = Periods(self.rtint, self.rampl, self.rprop)
        self.ptint, self.pampl, self.pprop = self.periods.all()
        self.opint, self.opamp, self.oppro = self.periods.open()
        self.shint, self.shamp, self.shpro = self.periods.shut()


class Periods:
    """ """
    def __init__(self, intervals, amplitudes, flags=None):
        self.rtint, self.rampl = intervals, amplitudes
        if flags is not None:
            self.rprop = flags
        else:
            self.rprop = np.zeros(len(intervals))
        self.ptint, self.pampl, self.pprop = [], [], []
        self._set_periods()
        
    def _set_periods(self):
        """ Separate the entire record into periods when channel is open or shut.
        There may be many amplitude transitions during one opening, each of 
        which will count as an individual opening, so generally better to
        look at 'open periods'. """
        
        # Remove first and last intervals if shut
        while self.rampl[0] == 0:
            self.rtint = self.rtint[1:]
            self.rampl = self.rampl[1:]
            self.rprop = self.rprop[1:]
        while (self.rtint[-1] < 0) or (self.rampl[-1] == 0):
            self.rtint = self.rtint[:-1]
            self.rampl = self.rampl[:-1]
            self.rprop = self.rprop[:-1]

        oint, oamp, oopt = self.rtint[0], self.rampl[0] * self.rtint[0], self.rprop[0]
        for t, a, o in zip(self.rtint[1 : ], self.rampl[1 : ], self.rprop[1 : ]):
            if o >= 8: oopt = 8
            condition_both_open = ((math.fabs(a) > 0.0) and (math.fabs(oamp) > 0.0))
            condition_both_shut = ((a == 0.0) and (oamp == 0.0))
            if condition_both_open or condition_both_shut:
                oint += t
                oamp += a * t
            else:
                try:
                    self.pampl.append(oamp / oint)
                except:
                    self.pampl.append(oamp)
                self.ptint.append(oint)
                self.pprop.append(oopt)
                oint, oamp, oopt = t, a, o
        # append last period
        try:
            self.pampl.append(oamp / oint)
        except:
            self.pampl.append(oamp)
        self.ptint.append(oint)
        self.pprop.append(oopt)

    def open(self):
        # TODO: remove bad intervals befor returning
        return self.ptint[0::2], self.pampl[0::2], self.pprop[0::2]

    def shut(self):
        # TODO: remove bad intervals befor returning
        return self.ptint[1::2], self.pampl[1::2], self.pprop[1::2]

    def all(self):
        # TODO: remove bad intervals befor returning
        return self.ptint, self.pampl, self.pprop


class Bursts(object):
    """   """
    def __init__(self, intervals, amplitudes, flags=None, tcrit=None):
        self.t, self.a = np.array(intervals), np.array(amplitudes)
        self.bursts = None
        if flags is not None:
            self.o = flags
        else:
            self.o = np.zeros(len(intervals))

    def slice_bursts(self, tcrit):
        """Cut entire single channel record into clusters using critical shut time
        interval (tcrit).
        Default definition of cluster:
        (1) Doesn't require a gap > tcrit before the 1st cluster in each record;
        (2) Unusable shut time is a valid end of cluster;
        (3) Open probability of a cluster is calculated without considering
        last opening. """

        self.tcrit = tcrit
        long_shuts = np.intersect1d(np.where(self.t > tcrit),
                                    np.where(self.a == 0.0))
        groups = np.split(self.t, long_shuts)
        #amplitudes = np.split(self.t, long_shuts)
        #bamps = [amplitudes[0]] + [np.delete(a, 0) for a in amplitudes[1:]]
        self.bursts = [groups[0]] + [np.delete(a, 0) for a in groups[1:]]
    
    def get_burst_total_open_time(self, burst):
        return np.sum(burst[0::2])

    def get_burst_popen1(self, burst):
        """ Calculate Popen by excluding very last opening. Equal number of open
        and shut intervals are taken into account. """
        if len(burst) > 1:
            return ((self.get_burst_total_open_time(burst) - burst[-1]) /
                np.sum(burst[:-1]))
        else:
            return 1.0

    def get_burst_popen(self, burst):
        """ Calculate burst Popen. """
        return self.get_burst_total_open_time(burst) / np.sum(burst)

    def get_list_popen(self):
        return [self.get_burst_popen1(b) for b in self.bursts]

    def get_list_length(self):
        return [np.sum(b) for b in self.bursts]

    def get_list_number_openings(self):
        return [len(b[0::2]) for b in self.bursts]

    def get_list_mean_opening_length(self):
        return [np.average(b[0::2]) for b in self.bursts]

    def get_list_total_open_time(self):
        return [np.sum(b[0::2]) for b in self.bursts]

    def get_list_total_shut_time(self):
        return [np.sum(b[1::2]) for b in self.bursts if len(b)>1]

    def get_mean_popen(self):
        Popen = self.get_list_popen()
        return np.average([x for x in Popen if str(x) != 'nan'])

    def get_mean_number_openings(self):
        return np.average(self.get_list_number_openings())

    def get_mean_length(self):
        return np.average(self.get_list_length())

    def get_bursts_with_opening_number(self, k):
        return [b for b in self.bursts if len(b[0::2]) == k]

    def remove_bursts_with_long_open_times(self, longop):
        """ Remove bursts which contain open intervals longer than specified 
        value. """
        return [b for b in self.bursts if b[0::2].any <= longop]

    def __repr__(self):
        ret_str = ('tcrit= {0:.3f} ms; number of bursts = {1:d}; '.
            format(self.tcrit * 1000, len(self.bursts)) +
            '\nmean length = {0:.6g} ms; '.format(self.get_mean_length() * 1000) +
            '\nmean Popen = {0:.3f}'.format(self.get_mean_popen()) +
            '\nmean number of openings = {0:.2f}'.format(self.get_mean_number_openings()))
        return ret_str