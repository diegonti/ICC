"""
Problem 3.1.3 - Audio Resampling
Modifying an audio file by its raw array data.
Diego Ontiveros
"""
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

factor = 100                                                # Modification factor, for a better representation, it has been exagerated to 100
samplerate, data = sp.io.wavfile.read("HartreeFock.wav")    # Gets the sample rate and data (array) of a sound file
new_samplerate = int(samplerate/factor)                     # New sample rate, divided by the reduction factor chosen

seconds = data.shape[0]/samplerate                          # Calculating the seconds the song lasts

time = np.arange(0,seconds, 1/samplerate)                   # Time array for the original file
time_resampled = np.arange(0,seconds, 1/samplerate*factor)  # Time array for the new file, since we want samplerate/100, the audio will have 100 times 
print(f"The audio file is {seconds} seconds long.")         # less points and the step from point to point will be multiplied (1/samplerate*100)
print(f"Old samplerate: {samplerate}Hz\nNew samplerate: {new_samplerate}Hz")

# Data treatement, reescaled amplitude by 1e4 to work in a better range (I believe they are arbitrary units)
if data.shape[1] == 2: # For stereo sound (2 channels, R/L, 2D array)

    dataL = data[:,0]/1e4   # Left channel data
    dataR = data[:,1]/1e4   # Right channel data

    dataL_resampled = dataL[::factor] # Resampling every 100 points (we remove a lot of the points, half if factor=2)
    dataR_resampled = dataR[::factor] # By changing the sample rate, we are also changing the sample depth

    # Plot Settings
    fig,axes = plt.subplots(2,1, figsize=(10,6))
    lw = 0.15 #Line width                                   # When plotting the audio data, the array for the x coordinate (time) for 
    axes[0].plot(time, dataL, lw=lw,                        # the resampled files has to have a step 100 times (*factor) the original one
        label = f"Original ({samplerate}Hz)")               # otherwhise with the same time coordinate (array) it would appear much shorter         
    axes[1].plot(time, dataR, lw=lw)                        # (since for each 100 points of the original file there's only one of the resampled one)    
    axes[0].plot(time_resampled, dataL_resampled, lw=lw, 
        label = f"Resampled ({new_samplerate}Hz)")    
    axes[1].plot(time_resampled, dataR_resampled, lw=lw)    
    
    axes[1].set_xlabel("time (s)")
    axes[0].set_ylabel(r"Amplitude (AU/$10^4$)");axes[1].set_ylabel(r"Amplitude (AU/$10^4$)")
    axes[0].legend(loc = "upper right")
    axes[0].set_title(f"Comparison of audiowaves with {samplerate}Hz and {new_samplerate}Hz samplerates.")
    axes[0].text(0.90, 0.05, "L Channel",transform=axes[0].transAxes)
    axes[1].text(0.90, 0.05, "R Channel",transform=axes[1].transAxes)

if data.shape[1] == 1: # For mono sound (1 channel, 1D array)
    
    data_resampled = data[::factor]/10000

    lw = 0.15
    fig,axes = plt.subplots(2,1, figsize=(8,8))
    axes[0].plot(time, data, lw=lw)
    axes[1].plot(time_resampled, data_resampled, lw=lw)

plt.show()

# If the factor of resampling is 2 (half), there's still a lot of loss of information, but since many points are still available, the
# soundwaves are almost the same. In here, a factor of 100 is chosen to better exemplify the loss of information on the wave when removing 
# that many data. The audio volume diminishes a lot and becomes almost understandable.

# To exemplify this, the resampled sound is converted back to raw audio data into an mp3 file, which with Audacity (importing as raw data) can be treated to produce a
# listenable file. The volume was increased and the exported as "HartreeFock_resampled.mp3"

# sp.io.wavfile.write("test.mp3", int(samplerate/factor), np.dstack((dataL_resampled, dataR_resampled))) #Exports raw audio data that can be treated with Audacity

