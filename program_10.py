#!/bin/env python
# Created on March 25, 2020
#  by Keith Cherkauer
#
# This script serves a as the solution set for assignment-10 on descriptive
# statistics and environmental informatics.  See the assignment documention 
# and repository at:
# https://github.com/Environmental-Informatics/assignment-10.git for more
# details about the assignment.

""" Program to perform Data Analysis
    based on various statistical 
    matrices on River-flow data
    
    Author: Varun Aggarwal
    Username: aggarw82
    Github: https://github.com/Environmental-Informatics/10-statistics-and-metrics-aggarw82
"""

import pandas as pd
import scipy.stats as stats
import numpy as np

def ReadData( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    raw data read from that file in a Pandas DataFrame.  The DataFrame index
    should be the year, month and day of the observation.  DataFrame headers
    should be "agency_cd", "site_no", "Date", "Discharge", "Quality". The 
    "Date" column should be used as the DataFrame index. The pandas read_csv
    function will automatically replace missing values with np.NaN, but needs
    help identifying other flags used by the USGS to indicate no data is 
    availabiel.  Function returns the completed DataFrame, and a dictionary 
    designed to contain all missing value counts that is initialized with
    days missing between the first and last date of the file."""
    
    # define column names
    colNames = ['agency_cd', 'site_no', 'Date', 'Discharge', 'Quality']

    # open and read the file
    DataDF = pd.read_csv(fileName, 
                         names=colNames,  
                         header=1, 
                         delimiter=r"\s+",parse_dates=[2], comment='#',
                         na_values=['Eqp'])
    DataDF = DataDF.set_index('Date')
    
    # To remove negative streamline values 
    for i in DataDF["Discharge"]:
        if i < 0: 
            DataDF["Discharge"][i] = np.NaN
    
    # quantify the number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
    
    # filtering out NoDate values 
    DataDF = DataDF.dropna()
    
    return( DataDF, MissingValues )

def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""
       
    # Clips the data to date range: startDate to endDate 
    DataDF = DataDF.loc[startDate:endDate]
    
    # quantify the number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
    
    return( DataDF, MissingValues )

def CalcTqmean(Qvalues):
    """This function computes the Tqmean of a series of data, typically
       a 1 year time series of streamflow, after filtering out NoData
       values.  Tqmean is the fraction of time that daily streamflow
       exceeds mean streamflow for each year. Tqmean is based on the
       duration rather than the volume of streamflow. The routine returns
       the Tqmean value for the given data array."""
    
    # filtering out NoDate values 
    Qvalues = Qvalues.dropna()
        
    # calculating mean streamflow for each year     
    Tqmean = ( (Qvalues > Qvalues.mean()).sum() ) / len(Qvalues)
    
    return ( Tqmean )

def CalcRBindex(Qvalues):
    """This function computes the Richards-Baker Flashiness Index
       (R-B Index) of an array of values, typically a 1 year time
       series of streamflow, after filtering out the NoData values.
       The index is calculated by dividing the sum of the absolute
       values of day-to-day changes in daily discharge volumes
       (pathlength) by total discharge volumes for each year. The
       routine returns the RBindex value for the given data array."""
    
    #filtering out NoData values
    Qvalues = Qvalues.dropna()
    
    # calculating day-to-day changes in discharge volumes
    abs_deltaQ = abs(Qvalues.diff()) 
    
    # calculating RBindex
    RBindex = abs_deltaQ.sum()/Qvalues.sum()
    
    return ( RBindex )

def Calc7Q(Qvalues):
    """This function computes the seven day low flow of an array of 
       values, typically a 1 year time series of streamflow, after 
       filtering out the NoData values. The index is calculated by 
       computing a 7-day moving average for the annual dataset, and 
       picking the lowest average flow in any 7-day period during
       that year.  The routine returns the 7Q (7-day low flow) value
       for the given data array."""
    
    #filtering out NoData values
    Qvalues = Qvalues.dropna()
    
    # calculating val7Q
    val7Q = Qvalues.rolling(window = 7).mean().min()
    
    return ( val7Q )

def CalcExceed3TimesMedian(Qvalues):
    """This function computes the number of days with flows greater 
       than 3 times the annual median flow. The index is calculated by 
       computing the median flow from the given dataset (or using the value
       provided) and then counting the number of days with flow greater than 
       3 times that value.   The routine returns the count of events greater 
       than 3 times the median annual flow value for the given data array."""
    
    #filtering out NoData values
    Qvalues = Qvalues.dropna()

    # calculating 3 times annual median flow
    m3x_value = 3 * Qvalues.median() 
    
    # adding values > 3*median
    median3x = 0 
    for i in range(len(Qvalues)):
        if Qvalues[i] > m3x_value:
            median3x += 1
    
    return ( median3x )

def GetAnnualStatistics(DataDF):
    """This function calculates annual descriptive statistcs and metrics for 
    the given streamflow time series.  Values are retuned as a dataframe of
    annual values for each water year.  Water year, as defined by the USGS,
    starts on October 1."""
    
    # naming all the columns
    columns = ['site_no','Mean Flow', 'Peak Flow','Median Flow','Coeff Var', 'Skew','Tqmean','R-B Index','7Q','3xMedian']
    
    # resampling the data 
    yearly_data = DataDF.resample('AS-OCT').mean()
    
    # defining new datafram
    WYDataDF = pd.DataFrame(0, 
                            index = yearly_data.index,
                            columns = columns)
    
    # resampling data for water year 
    ydf = DataDF.resample('AS-OCT')
    
    # calculating different annual statistics 
    WYDataDF['site_no'] = ydf['site_no'].mean()
    WYDataDF['Mean Flow'] = ydf['Discharge'].mean()
    WYDataDF['Peak Flow'] = ydf['Discharge'].max()
    WYDataDF['Median Flow'] = ydf['Discharge'].median()
    WYDataDF['Coeff Var'] = (ydf['Discharge'].std()/ydf['Discharge'].mean())*100
    WYDataDF['Skew'] = ydf['Discharge'].apply(lambda x: stats.skew(x))
    WYDataDF['Tqmean'] = ydf.apply({'Discharge': lambda x: CalcTqmean(x)})
    WYDataDF['R-B Index'] = ydf.apply({'Discharge':lambda x: CalcRBindex(x)})
    WYDataDF['7Q'] = ydf.apply({'Discharge':lambda x: Calc7Q(x)})
    WYDataDF['3xMedian'] = ydf.apply({'Discharge':lambda x: CalcExceed3TimesMedian(x)})
    
    return ( WYDataDF )

def GetMonthlyStatistics(DataDF):
    """This function calculates monthly descriptive statistics and metrics 
    for the given streamflow time series.  Values are returned as a dataframe
    of monthly values for each year."""

    # names of columns
    columns = ['site_no', 'Mean Flow', 'Coeff Var', 'Tqmean', 'R-B Index']
    
    # resampling data
    monthly_data = DataDF.resample('BMS').mean()
    
    # saving monthly data
    MoDataDF = pd.DataFrame(index = monthly_data.index, columns = columns) 
    
    # defining new dataframe for monthly statistics
    mdf = DataDF.resample('BMS') 
    
    #Calculating monthly statistics as described in instructions
    MoDataDF['site_no']=mdf['site_no'].mean()
    MoDataDF['Mean Flow'] = mdf['Discharge'].mean()
    MoDataDF['Coeff Var'] = (mdf['Discharge'].std() / mdf['Discharge'].mean()) * 100
    MoDataDF['Tqmean'] = mdf.apply({'Discharge':lambda x: CalcTqmean(x)})
    MoDataDF['R-B Index'] = mdf.apply({'Discharge':lambda x: CalcRBindex(x)})

    return ( MoDataDF )

def GetAnnualAverages(WYDataDF):
    """This function calculates annual average values for all statistics and
    metrics.  The routine returns an array of mean values for each metric
    in the original dataframe."""
    
    # annual average calculation
    AnnualAverages=WYDataDF.mean(axis=0)
    
    return( AnnualAverages )

def GetMonthlyAverages(MoDataDF):
    """This function calculates annual average monthly values for all 
    statistics and metrics.  The routine returns an array of mean values 
    for each metric in the original dataframe."""
    
    # defining months and column names 
    month = [3,4,5,6,7,8,9,10,11,0,1,2]
    columns = ['site_no', 'Mean Flow', 'Coeff Var', 'Tqmean', 'R-B Index']

    # creating dataframe to store monthly statistics 
    MonthlyAverages = pd.DataFrame( 0, index=range(1, 13), 
                                       columns = columns)
    

    for i in range(12):
        MonthlyAverages.iloc[i,0]=MoDataDF['site_no'][::12].mean()
        MonthlyAverages.iloc[i,1]=MoDataDF['Mean Flow'][month[i]::12].mean()
        MonthlyAverages.iloc[i,2]=MoDataDF['Coeff Var'][month[i]::12].mean()
        MonthlyAverages.iloc[i,3]=MoDataDF['Tqmean'][month[i]::12].mean()
        MonthlyAverages.iloc[i,4]=MoDataDF['R-B Index'][month[i]::12].mean()
    
    return( MonthlyAverages )

# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':

    # define filenames as a dictionary
    # NOTE - you could include more than jsut the filename in a dictionary, 
    #  such as full name of the river or gaging site, units, etc. that would
    #  be used later in the program, like when plotting the data.
    fileName = { "Wildcat": "WildcatCreek_Discharge_03335000_19540601-20200315.txt",
                 "Tippe": "TippecanoeRiver_Discharge_03331500_19431001-20200315.txt" }
    
    # define blank dictionaries (these will use the same keys as fileName)
    DataDF = {}
    MissingValues = {}
    WYDataDF = {}
    MoDataDF = {}
    AnnualAverages = {}
    MonthlyAverages = {}
    
    # process input datasets
    for file in fileName.keys():
        
        print( "\n", "="*50, "\n  Working on {} \n".format(file), "="*50, "\n" )
        
        DataDF[file], MissingValues[file] = ReadData(fileName[file])
        print( "-"*50, "\n\nRaw data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # clip to consistent period
        DataDF[file], MissingValues[file] = ClipData( DataDF[file], '1969-10-01', '2019-09-30' )
        print( "-"*50, "\n\nSelected period data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # calculate descriptive statistics for each water year
        WYDataDF[file] = GetAnnualStatistics(DataDF[file])
        
        # calcualte the annual average for each stistic or metric
        AnnualAverages[file] = GetAnnualAverages(WYDataDF[file])
        
        print("-"*50, "\n\nSummary of water year metrics...\n\n", WYDataDF[file].describe(), "\n\nAnnual water year averages...\n\n", AnnualAverages[file])

        # calculate descriptive statistics for each month
        MoDataDF[file] = GetMonthlyStatistics(DataDF[file])

        # calculate the annual averages for each statistics on a monthly basis
        MonthlyAverages[file] = GetMonthlyAverages(MoDataDF[file])
        
        print("-"*50, "\n\nSummary of monthly metrics...\n\n", MoDataDF[file].describe(), "\n\nAnnual Monthly Averages...\n\n", MonthlyAverages[file])
                
    # saving annual statistics  
    # Wildcate data
    wildcat = WYDataDF['Wildcat']
    wildcat['Station'] = 'Wildcat'
    # Tippe data
    tippe = WYDataDF['Tippe']
    tippe['Station'] = 'Tippe'
    # combining the two data
    both = wildcat.append(tippe)
    both.to_csv('Annual_Metrics.csv',sep=',', index=True)
        
    # saving monthly statistics
    # wildcat data 
    wildcat_mon = MoDataDF['Wildcat']
    wildcat_mon['Station'] = 'Wildcat'
    # Tippe data
    tippe_mon = MoDataDF['Tippe']
    tippe_mon['Station'] = 'Tippe'
    # combining both the data
    both_mon = wildcat_mon.append(tippe_mon)
    both_mon.to_csv('Monthly_Metrics.csv',sep=',', index=True)
    
    # saving annual average statistics 
    # Wildcat data
    wildcat_anavg = AnnualAverages['Wildcat']
    wildcat_anavg['Station'] = 'Wildcat'
    # Tippe data
    tippe_anavg = AnnualAverages['Tippe']
    tippe_anavg['Station'] = 'Tippe'
    # combining the data
    both_anavg = wildcat_anavg.append(tippe_anavg)
    both_anavg.to_csv('Average_Annual_Metrics.txt',sep='\t', index=True)
        
    # saving monthly average statistics 
    # Wildcat data 
    wildcat_moavg = MonthlyAverages['Wildcat']
    wildcat_moavg['Station'] = 'Wildcat'
    # Tippe data
    tippe_moavg = MonthlyAverages['Tippe']
    tippe_moavg['Station'] = 'Tippe'
    # combining the data
    both_moavg = wildcat_moavg.append(tippe_moavg)
    both_moavg.to_csv('Average_Monthly_Metrics.txt',sep='\t', index=True)

