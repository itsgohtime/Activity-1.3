import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

# Splits the files into periods of each mode 
def Find_Segments(FileName):
  Data=pd.read_csv(FileName)
  # Finds the value of the first mode 
  mode = Data['Mode'][0]
  start_i = 0; periods = []

  # Loops through the data and finds when the mode changes and then 
  # appends the last starting index and the end index of that period
  for i in range(len(Data)):
      if Data['Mode'][i] != mode:
        end_i = i - 1
        periods.append([start_i, end_i, mode])
        start_i = i; mode = Data['Mode'][i] 
      # Ignores the last period if it's not standing
      elif i == len(Data) - 1 and mode == 1:
        periods.append([start_i, i, mode])
  return Data, periods

# Finds the mean value of the knee angle for each period
def F1 (Data, period):
  knee_angle =  Data['Knee_Angle'][period[0]: period[1] + 1]
  return knee_angle.mean()

# Finds the integral of the PSD
def F2 (Data, period):
  # Gets the time, Sampling time, number of points and segments, and knee velocity
  time = Data['time'][period[0]: period[1] + 1]
  SampTime = Data['time'][1] - Data['time'][0] 
  nPts = time.size
  nSegments = 1
  knee_vel =  Data['Knee_Velocity'][period[0]: period[1] + 1]

  # Converts to the frequency domain
  frequency, PowerSpectrum = signal.welch(knee_vel,1/SampTime,nperseg=np.int32(nPts/nSegments))

  # Interpolates the data so that the integral can be taken
  f = InterpolatedUnivariateSpline(frequency, PowerSpectrum, k=1) 
  return f.integral(frequency[0], frequency[-1])

# Splits the data based on the number of peaks
def Split(Data, period):

  # Finds the number of peaks above 0.2 in the knee velocity data
  knee_vel =  Data['Knee_Velocity'][period[0]: period[1] + 1]
  peaks = signal.find_peaks(knee_vel, height = 0.2)[0]

  # Finds the number of points in each section
  if len(peaks)!= 0:
    n = int(len(knee_vel)/len(peaks))
  else:
    n = len(knee_vel)

  # Splits the data based on the number of points in each section
  final = [knee_vel[i * n:(i + 1) * n] for i in range((len(knee_vel) + n - 1) // n )]
  return final

# Finds the shape factor of each period
def F3 (Data, period):
  shape_factor = []

  # Splits the data based on the number of peaks
  Data_New=Split(Data,period)

  # Finds the shape factor for each peak and then takes the mean
  for j in range(len(Data_New)):
      rms=np.sqrt(np.mean(Data_New[j]**2))
      abs_mean=np.mean(abs(Data_New[j]))
      shape_factor.append(rms/abs_mean)
  sf_mean = sum(shape_factor)/len(shape_factor)
  return sf_mean

def ComputeWithinModeFeatures(Data, period):

  #feature 1: knee_velocity mean
  f1= F1(Data, period)          #feature
  f2=F2(Data, period)           #feature
  f3=F3(Data, period)           #feature
  feature_vals=[f1,f2, f3]
    
    #Create holders to store names and values of features. Fill with mode info  
  FeatureNames=['Mode', "knee_velocity_mean", "knee_velocity_psd", "knee_velocity_shape_factor"]
  FeatureValues=np.array([[period[2]]])
    
  #iterate through series, creating features for each
  #return feature_vals
  FeatureValues=np.append(FeatureValues,feature_vals)
  
  Features=pd.DataFrame(data=np.array([FeatureValues]),columns=FeatureNames)
  return Features

# Finds the feature for each period for each file and makes it into a database
def FindFeatures(FileNames):
  Features = pd.DataFrame()
  for File in FileNames:
      Data, periods = Find_Segments(File)
      for i in periods:
        New_Features = ComputeWithinModeFeatures(Data, i)
        Features=pd.concat([Features, New_Features],ignore_index=True)
  return Features
