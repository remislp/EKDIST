import os
import numpy as np

from ekdist import ekscn
from ekdist import ekrecord

class TestSCNFileLoading:
    def setUp(self):
        self.infile = "./tests/AChsim.scn"
        self.header = ekscn.read_header(self.infile)
        self.itint, self.iampl, self.iprops = ekscn.read_data(self.infile, self.header)
        
    def test_infile_exists(self):
        assert os.path.isfile(self.infile)
        
    def test_SCN_header_loading(self):
        assert self.header
        
    def test_interval_number(self):
        assert self.header['nint'] == 13948
        
    def test_interval_loading(self):    
        assert self.header['nint'] == len(self.itint)
        
    def test_flags(self):
        assert not self.iprops.any()
        
    def test_amplitudes(self):
        assert self.iampl[0] == 0.
        assert self.iampl[1] == 6.
        assert self.iampl[2] == 0.

    def tearDown(self):
        self.header = None
        self.itint, self.iampl, self.iprops = None, None, None
        
class TestIntervalListLoading:
    def setUp(self):
        self.intervals = [20.0, 1.0, 19.0, 100.0, 10.0, 100.0, 1.0]
        self.amplitudes = [5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0]
        self.rec = ekrecord.SingleChannelRecord()
        self.rec.load_intervals_from_list(np.array(self.intervals), np.array(self.amplitudes))

    def test_original_number_intervals(self):
        assert len(self.rec.itint) == 7
        assert len(self.rec.iampl) == 7
        assert len(self.rec.iprop) == 7

    def test_imposing_resolution(self):
        self.rec.tres = 2.0
        assert len(self.rec.rtint) == 5
        assert len(self.rec.rampl) == 5
        assert len(self.rec.rprop) == 5

    def test_setting_periods(self):
        self.rec.tres = 2.0
        per = ekrecord.Periods(self.rec.rtint, self.rec.rampl, self.rec.rprop)
        assert len(per.ptint) == 3
        assert len(per.pampl) == 3
        assert len(per.pprop) == 3 

    def test_burst_number(self):
        self.rec.tres = 2.0
        per = ekrecord.Periods(self.rec.rtint, self.rec.rampl, self.rec.rprop)
        br = ekrecord.Bursts(per.ptint, per.pampl)
        br.slice_bursts(50.0)
        assert len(br.bursts) == 2 

    def tearDown(self):
        self.intervals, self.amplitudes = None, None
        self.rec = None
