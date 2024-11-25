# Here, we perform parameter estimation using bilby to get the masses and the sky location of the binary source

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import bilby
from bilby.core.prior import Uniform, Cosine
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters, generate_all_bbh_parameters

from gwpy.timeseries import TimeSeries

L1_data = TimeSeries.read('L1_data.txt')
H1_data = TimeSeries.read('H1_data.txt')
V1_data = TimeSeries.read('V1_data.txt')

print('Data file has been read')


H1 = bilby.gw.detector.get_empty_interferometer("H1")
L1 = bilby.gw.detector.get_empty_interferometer("L1")
V1 = bilby.gw.detector.get_empty_interferometer("V1")

print('interferometers initiated')


H1.set_strain_data_from_gwpy_timeseries(H1_data)
L1.set_strain_data_from_gwpy_timeseries(L1_data)
V1.set_strain_data_from_gwpy_timeseries(V1_data)

#from matched filtering

event_time = 1126259642.4199219


# setting up the prior
print('setting prior')

prior = bilby.core.prior.PriorDict()
prior['mass_1'] = Uniform(name='mass_1', minimum=20.0,maximum=50.0)
prior['mass_2'] = Uniform(name='mass_2', minimum=20.0, maximum=50.0)
prior['phase'] = 1.3
prior['geocent_time'] = Uniform(name="geocent_time", minimum=event_time-0.1, maximum=event_time+0.1)
prior['a_1'] =  0.0
prior['a_2'] =  0.0
prior['tilt_1'] =  0.0
prior['tilt_2'] =  0.0
prior['phi_12'] =  0.0
prior['phi_jl'] =  0.0
prior['dec'] = Cosine(name='dec')
prior['ra'] =  Uniform(name='ra', minimum=0, maximum=2*np.pi, boundary='periodic')
prior['theta_jn'] =  0.4
prior['psi'] =  0.0
prior['luminosity_distance'] = 900

# First, put our "data" created above into a list of intererometers (the order is arbitrary)
interferometers = [H1, L1, V1]

# Next create a dictionary of arguments which we pass into the LALSimulation waveform - we specify the waveform approximant here
waveform_arguments = dict(
    waveform_approximant='IMRPhenomPv2', reference_frequency=100., catch_waveform_errors=True)

# Next, create a waveform_generator object. This wraps up some of the jobs of converting between parameters etc
waveform_generator = bilby.gw.WaveformGenerator(
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    waveform_arguments=waveform_arguments,
    parameter_conversion=convert_to_lal_binary_black_hole_parameters)

# Finally, create our likelihood, passing in what is needed to get going
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(interferometers, waveform_generator, priors=prior, time_marginalization=True)

print('estimating parameters')
result_short = bilby.run_sampler(likelihood, prior, sampler='dynesty', outdir='short', label="fake_data",conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,nlive=1024, dlogz=0.1,clean=True,)
print(result_short.posterior)
print('saving posterior dataframe')

# saving results to posterior.csv
df = result_short.posterior

df.to_csv('posterior.csv')