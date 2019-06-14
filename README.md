# EKDIST
Explore idealised single channel records and fit many sorts of distributions (eg exponential pdf to dwell times, gaussian to amplitudes).

Idealisation of an experimental single channel record produces a list of durations and amplitudes of each single channel event (opening or shutting).  EKDIST package provides scripts/functions to view and/or fit these events.

## Imposing resolution 

## Stability plots
Stability plots display the measurments made at equilibrium against the measurment number (suggested by Weiss and Magleby, 1989). It allows to check the data for stability before distributions are constructed or fitted. If changes in measurment occur with time the corresponding distribution is meaningless. 

#### Stability plot for amplitudes 
This plots all the amplitudes as dots against event number.  Such a plot allows to see whether the various amplitude levels stay constant throughout the experiment. Each amplitude by default is shown individually, though it is possible to display a running average of several amplitudes.  Example plot below shows stability plots for amplitudes of single-channel openings of glycine receptor activated by 10 &mu;M of glycine. 

![Amplitude stability](/images/stability_amplitudes.png)
 
#### Stability plots for open periods, shut times or P(open) 
In stability plot the open times and shut times, being very variable, are averaged in groups before plotting against event number.  The defaults are to calculate running averages of 50 intervals.  The mean open and shut time, and the Popen calculated from them, are plotted (together or separately) on a log scale against event number.
Example plot below shows stability plots for open times, shut times and Popen from single-channel record of glycine receptor activated by 10 &mu;M of glycine. 

![Openings/shuttings/Popen stability](/images/stability_open_shut_Popen.png)

## Amplitudes

#### All point histogram
In all point amplitude histogram, the contribution (area) of each opening (or shutting) is proportional to its length (longer intervals have more data points).

#### Histogram of measured amplitudes
In a histogram of measured amplitudes each opening contributes one value to the histogram, regardless of the length of the opening.  

![Fitted amplitudes histogram with Gaussian fit](/images/histogram_fitted_amplitudes.png)

## Dwell time distributions

## Bursts

# Installation

If you are using 'pip' then the `ekdist` package can be installed locally with (note a space and a dot at the end):

`pip install .`

or

`pip install -e .`

The latter creates symlink which allows any source code change be available locally immediately.


# Ongoing development
This software is under development. Output is not fully tested, and therefore you should exercise caution if you plan to use the results for production / publication. 

Expected additions: 
- fit amplitude histogram with Gaussian multiple components;
- plot dwell time distribbutions and fit with exponential pdf's;
- burst analysis;
- GUI;
- tests.

