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
        self.iprop[self.itint < 0] = 8
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
        """
        Separate open and shut intervals from the entire record.
        There may be many small amplitude transitions during one opening,
        each of which will count as an individual opening, so generally
        better to look at 'open periods'.
        Look for start of a group of openings i.e. any opening that has
        defined duration (i.e. usable).  A single unusable opening in a group
        makes its length undefined so it is excluded.
        NEW VERSION -ENSURES EACH OPEN PERIOD STARTS WITH SHUT-OPEN TRANSITION
        Find start of a group (open period) -valid start must have a good shut
        time followed by a good opening -if a bad opening is found as first (or
        any later) opening then the open period is abandoned altogether, and the
        next good shut time sought as start for next open period, but for the
        purposes of identifying the nth open period, rejected ones must be counted
        as an open period even though their length is undefined.
        """

        pint, pamp, popt = [], [], []
        # Remove first and last intervals if shut
        if self.rampl[0] == 0:
            self.rtint = self.rtint[1:]
            self.rampl = self.rampl[1:]
            self.rprop = self.rprop[1:]
        if self.rtint[-1] < 0:
            self.rtint = self.rtint[:-1]
            self.rampl = self.rampl[:-1]
            self.rprop = self.rprop[:-1]
        while self.rampl[-1] == 0:
            self.rtint = self.rtint[:-1]
            self.rampl = self.rampl[:-1]
            self.rprop = self.rprop[:-1]

        oint, oamp, oopt = self.rtint[0], self.rampl[0] * self.rtint[0], self.rprop[0]
        n = 1
        while n < len(self.rtint):
            if self.rampl[n] != 0:
                oint += self.rtint[n]
                oamp += self.rampl[n] * self.rtint[n]
                if self.rprop[n] >= 8: oopt = 8

                if n == (len(self.rtint) - 1):
                    pamp.append(oamp/oint)
                    pint.append(oint)
                    popt.append(oopt)
            else:
                # found two consequent gaps
                if oamp == 0 and self.rampl[n] == 0 and oopt < 8:
                    pint[-1] += self.rtint[n]
                # skip bad opening
                #elif (self.badopen > 0 and oint > self.badopen) or (oopt >= 8):
                elif (oopt >= 8):
                    popt[-1] = 8
                    oint, oamp, oopt = 0.0, 0.0, 0
#                    if n != (len(self.rint) - 2):
#                        n += 1
                else: # shutting terminates good opening
                    pamp.append(oamp/oint)
                    pint.append(oint)
                    popt.append(oopt)
                    oint, oamp, oopt = 0.0, 0.0, 0
                    pamp.append(0.0)
                    pint.append(self.rtint[n])
                    popt.append(self.rprop[n])
            n += 1

        self.ptint, self.pampl, self.pprop = pint, pamp, popt
        self.opint = self.ptint[0::2]
        self.opamp = self.pampl[0::2]
        self.oppro = self.pprop[0::2]
        self.shint = self.ptint[1::2]
        self.shamp = self.pampl[1::2]
        self.shpro = self.pprop[1::2]


    def get_bursts(self, tcrit):
        """
        Cut entire single channel record into clusters using critical shut time
        interval (tcrit).
        Default definition of cluster:
        (1) Doesn't require a gap > tcrit before the 1st cluster in each record;
        (2) Unusable shut time is a valid end of cluster;
        (3) Open probability of a cluster is calculated without considering
        last opening.
        """
        
        tcrit = math.fabs(tcrit)
        bursts = Bursts(tcrit)
        burst = Burst()
        i = 0
        badend = False
        while i < (len(self.ptint) - 1):
            if self.pampl[i] != 0:
                if not badend:
                    if self.pprop[i] < 8:
                        burst.add_interval(self.ptint[i], self.pampl[i])
                    else:
                        badend = True
            else: # found gap
                if ((self.ptint[i] < tcrit) and (self.pprop[i] < 8)):
                    if not badend:
                        burst.add_interval(self.ptint[i], self.pampl[i])
#                elif self.pint[i] >= tcrit and self.popt[i] < 8:
                else:
                    if ((burst.get_openings_number() > 0) and 
                        (burst.get_openings_number() * 2 == len(burst.intervals) + 1) 
                        and (not badend)):
                        bursts.add_burst(burst)
                    burst = Burst()
                    badend = False
#                elif self.popt[i] >= 8:
#                    badend = True
            i += 1
        if self.pampl[i] != 0:
            burst.add_interval(self.ptint[i], self.pampl[i])
            bursts.add_burst(burst)
        if burst.intervals and not badend:
            bursts.add_burst(burst)
        return bursts

class Burst(object):
    """   """
    def __init__(self):
        self.setup()

    def setup(self):
        self.intervals = []
        self.amplitudes = []

    def add_interval(self, interval, amplitude):
        self.intervals.append(interval)
        self.amplitudes.append(amplitude)

    def concatenate_last(self, interval, amplitude):
        try:
            self.intervals[-1] += interval
        except:
            self.intervals.append(interval)
            self.amplitudes.append(amplitude)

    def get_open_intervals(self):
        return self.intervals[0::2]

    def get_shut_intervals(self):
        return self.intervals[1::2]

    def get_mean_amplitude(self):
        return np.average(self.amplitudes[0::2])

    def get_openings_number(self):
        return len(self.get_open_intervals())

    def get_openings_average_length(self):
        return np.average(self.get_open_intervals())

    def get_shuttings_average_length(self):
        return np.average(self.get_shut_intervals())

    def get_total_open_time(self):
        return np.sum(self.get_open_intervals())

    def get_total_shut_time(self):
        return np.sum(self.get_shut_intervals())

    def get_length(self):
        return np.sum(self.intervals)

    def get_popen(self):
        """ Calculate burst Popen. """
        return self.get_total_open_time() / np.sum(self.intervals)

    def get_popen1(self):
        """ Calculate Popen by excluding very last opening. Equal number of open
        and shut intervals are taken into account. """
        if len(self.intervals) > 1:
            return ((self.get_total_open_time() - self.intervals[-1]) /
                np.sum(self.intervals[:-1]))
        else:
            return 1.0

    def get_running_mean_popen(self, N):
        if len(self.intervals)-1 > 2*N:
            openings = self.get_open_intervals()
            shuttings = self.get_shut_intervals()
            meanP = []
            for i in range(len(openings) - N):
                meanP.append(np.sum(openings[i: i+N]) /
                    (np.sum(openings[i: i+N]) + np.sum(shuttings[i: i+N])))
            return meanP
        else:
            return self.get_popen()

    def __repr__(self):
        ret_str = ('Group length = {0:.3f} ms; '.
            format(self.get_length() * 1000) +
            'number of openings = {0:d}; '.format(self.get_openings_number()) +
            'Popen = {0:.3f}'.format(self.get_popen()))
        if self.get_openings_number() > 1:
            ret_str += ('\n\t(Popen omitting last opening = {0:.3f})'.
            format(self.get_popen1()))
        ret_str += ('\n\tTotal open = {0:.3f} ms; total shut = {1:.3f} ms'.
            format(self.get_total_open_time() * 1000,
            self.get_total_shut_time() * 1000))
        return ret_str

class Bursts(object):
    """   """
    def __init__(self, tcrit):
        self.tcrit = tcrit
        self.bursts = []
    def add_burst(self, burst):
        self.bursts.append(burst)

    def intervals(self):
        """ Get all intervals in the record. """
        return [b.intervals for b in self.bursts]

    def get_op_lists(self):
        list = []
        for b in self.bursts:
            list.append(b.get_open_intervals())
        return list

    def get_sh_lists(self):
        list = []
        for b in self.bursts:
            list.append(b.get_shut_intervals())
        return list

    def all(self):
        return self.bursts

    def count(self):
        return len(self.bursts)

    def get_length_list(self):
        return [b.get_length() for b in self.bursts]

    def get_length_mean(self):
        return np.average(self.get_length_list())

    def get_opening_num_list(self):
        return [b.get_openings_number() for b in self.bursts]

    def get_opening_num_mean(self):
        return np.average(self.get_opening_num_list())

    def get_opening_length_mean_list(self):
        return [np.average(b.get_open_intervals()) for b in self.bursts]
    
    def get_popen_list(self):
        return [b.get_popen1() for b in self.bursts]

    def get_mean_ampl_list(self):
        return [b.get_mean_amplitude() for b in self.bursts]
    
    def get_popen_mean(self):
        Popen = self.get_popen_list()
        return np.average([x for x in Popen if str(x) != 'nan'])

    def get_long(self, minop):
        long = Bursts(self.tcrit)
        for b in self.bursts:
            if b.get_openings_number() >= minop:
                long.add_burst(b)
        return long
    
    def remove_long_open_times(self, top):
        """ Remove bursts which contain open intervals longer than specified 
        value. """
        cleaned = Bursts(self.tcrit)
        for b in self.bursts:
            if max(b.get_open_intervals()) <= top:
                cleaned.add_burst(b)
        return [b.intervals for b in cleaned.bursts]

    def __repr__(self):
        ret_str = ('tcrit= {0:.3f} ms; number of bursts = {1:d}; '.
            format(self.tcrit * 1000, len(self.bursts)) +
            '\nmean length = {0:.6g} ms; '.format(self.get_length_mean() *1000) +
            '\nmean Popen = {0:.3f}'.format(self.get_popen_mean()) +
            '\nmean number of openings = {0:.2f}'.format(self.get_opening_num_mean()))
        return ret_str
