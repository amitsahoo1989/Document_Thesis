
import pandas as pd

####################################################################################
fire_data=r'C:\Users\toshiba\Desktop\fire forest\Data1\Alaska.csv'
df1=pd.read_csv(fire_data)

#print(df1.isnull().sum())

df1=df1.drop(['Staffed Fires','Active Fires','LightningAcres','LightningFires','HumanAcres','HumanFires','Day'],axis=1)
#print(df1.head())
###################################################################################
weather_data_day0=r'C:\Users\toshiba\Desktop\fire forest\Data1\day0.csv'
df2=pd.read_csv(weather_data_day0)

##checking for null values
#print(df2.isnull().sum())

##droping the remarks column
df2=df2.drop(['URL','Remarks'],axis=1)

##Renaming the columns
df2.rename(columns={#'URL': 'date', 
                    'F_min': 'day0_F_min','F_mean':'day0_F_mean',
                    'F_max':'day0_F_max','IN_Sea_level':'day0_IN_Sea_level',
                    'F_Mean_dew_point':'day0_F_Mean_dew_point',
                    'IN_tot_rain':'day0_IN_tot_rain',
                    'MI_visibilty':'day0_MI_visibilty',
                    'MPH_mean_wind_speed':'day0_MPH_mean_wind_speed',
                    'MPH_max_sustained_wind':'day0_MPH_max_sustained_wind',
                    'MPH_max_wind':'day0_MPH_max_wind'}, inplace=True)
#print(df2.head())
###################################################################################
###################################################################################
weather_data_day1=r'C:\Users\toshiba\Desktop\fire forest\Data1\day1.csv'
df3=pd.read_csv(weather_data_day1)

##checking for null values
#print (df3.isnull().sum())

##droping the remarks column
df3=df3.drop(['URL','Remarks'],axis=1)

##Renaming the columns
df3.rename(columns={#'URL': 'date', 
                    'F_min': 'day1_F_min','F_mean':'day1_F_mean',
                    'F_max':'day1_F_max','IN_Sea_level':'day1_IN_Sea_level',
                    'F_Mean_dew_point':'day1_F_Mean_dew_point',
                    'IN_tot_rain':'day1_IN_tot_rain',
                    'MI_visibilty':'day1_MI_visibilty',
                    'MPH_mean_wind_speed':'day1_MPH_mean_wind_speed',
                    'MPH_max_sustained_wind':'day1_MPH_max_sustained_wind',
                    'MPH_max_wind':'day1_MPH_max_wind'}, inplace=True)

#################################################################################
###################################################################################
weather_data_day2=r'C:\Users\toshiba\Desktop\fire forest\Data1\day2.csv'
df4=pd.read_csv(weather_data_day2)



##droping the remarks column
df4=df4.drop(['URL','Remarks','Snow_depth'],axis=1)

##checking for null values
#print (df4.isnull().sum())


##Renaming the columns
df4.rename(columns={'F_min': 'day2_F_min','F_mean':'day2_F_mean',
                    'F_max':'day2_F_max','IN_Sea_level':'day2_IN_Sea_level',
                    'F_Mean_dew_point':'day2_F_Mean_dew_point',
                    'IN_tot_rain':'day2_IN_tot_rain',
                    'MI_visibilty':'day2_MI_visibilty',
                    'MPH_mean_wind_speed':'day2_MPH_mean_wind_speed',
                    'MPH_max_sustained_wind':'day2_MPH_max_sustained_wind',
                    'MPH_max_wind':'day2_MPH_max_wind'}, inplace=True)
#print(df4.head())
#################################################################################

###################################################################################
weather_data_day4=r'C:\Users\toshiba\Desktop\fire forest\Data1\day3.csv'
df5=pd.read_csv(weather_data_day4)


##droping the remarks column
df5=df5.drop(['URL','Remarks','Snow_depth'],axis=1)

##checking for null values
#print (df5.isnull().sum())


##Renaming the columns
df5.rename(columns={'F_min': 'day3_F_min','F_mean':'day3_F_mean',
                    'F_max':'day3_F_max','IN_Sea_level':'day3_IN_Sea_level',
                    'F_Mean_dew_point':'day3_F_Mean_dew_point',
                    'IN_tot_rain':'day3_IN_tot_rain',
                    'MI_visibilty':'day3_MI_visibilty',
                    'MPH_mean_wind_speed':'day3_MPH_mean_wind_speed',
                    'MPH_max_sustained_wind':'day3_MPH_max_sustained_wind',
                    'MPH_max_wind':'day3_MPH_max_wind'}, inplace=True)
#print(df5.head())
#################################################################################
###################################################################################
weather_data_day5=r'C:\Users\toshiba\Desktop\fire forest\Data1\day4.csv'
df6=pd.read_csv(weather_data_day5)

##droping the remarks column
df6=df6.drop(['URL','Remarks','Snow_depth'],axis=1)

##checking for null values
#print (df6.isnull().sum())


##Renaming the columns
df6.rename(columns={'F_min': 'day4_F_min','F_mean':'day4_F_mean',
                    'F_max':'day4_F_max','IN_Sea_level':'day4_IN_Sea_level',
                    'F_Mean_dew_point':'day4_F_Mean_dew_point',
                    'IN_tot_rain':'day4_IN_tot_rain',
                    'MI_visibilty':'day4_MI_visibilty',
                    'MPH_mean_wind_speed':'day4_MPH_mean_wind_speed',
                    'MPH_max_sustained_wind':'day4_MPH_max_sustained_wind',
                    'MPH_max_wind':'day4_MPH_max_wind'}, inplace=True)
#print(df6.head())
#############################################################################
result = pd.concat([df1, df2,df3,df4,df5,df6], axis=1, join='inner')

print(result.columns)

result.to_csv(r'C:\Users\toshiba\Desktop\fire forest\Data1\fire_weather_final.csv',index=False)