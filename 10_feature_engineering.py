import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def histogram_plot(dataset, title):
    plt.figure(figsize=(8, 6))    
    
    ax = plt.subplot()    
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)
    
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left() 
    
    plt.title(title, fontsize = 22)
    plt.hist(dataset, edgecolor='black', linewidth=1.8)
    
    
file=r'C:\Users\toshiba\Desktop\fire forest\Data1\mice_imputed.csv'
df=pd.read_csv(file)

df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
print(df.dtypes)


df[['FireSeason','Month','PrepLevel','Fire_Severity']] = df[['FireSeason','Month','PrepLevel','Fire_Severity']].astype(
        'category',copy=False)

for i in df.columns.values.tolist():
    histogram_plot(df[i],i)
    

histogram_plot(df['day0_MI_visibilty'], title = "Visibilty of day0")
plt.show()
df['log_day0_MI_visibilty'] = np.log(1 + df['day0_MI_visibilty'])
histogram_plot(df['log_day0_MI_visibilty'], title = "Visibilty of day0 in log")
plt.show()


histogram_plot(df['day1_MI_visibilty'], title = "Visibilty of day1")
plt.show()
df['log_day1_MI_visibilty'] = np.log(1 + df['day1_MI_visibilty'])
histogram_plot(df['log_day1_MI_visibilty'], title = "Visibilty of day1 in log")
plt.show()

histogram_plot(df['day2_MI_visibilty'], title = "Visibilty of day2")
plt.show()
df['log_day2_MI_visibilty'] = np.log(1 + df['day2_MI_visibilty'])
histogram_plot(df['log_day2_MI_visibilty'], title = "Visibilty of day2 in log")
plt.show()


histogram_plot(df['day3_MI_visibilty'], title = "Visibilty of day3")
plt.show()
df['log_day3_MI_visibilty'] = np.log(1 + df['day3_MI_visibilty'])
histogram_plot(df['log_day3_MI_visibilty'], title = "Visibilty of day3 in log")
plt.show()


histogram_plot(df['day4_MI_visibilty'], title = "Visibilty of day4")
plt.show()
df['log_day4_MI_visibilty'] = np.log(1 + df['day4_MI_visibilty'])
histogram_plot(df['log_day4_MI_visibilty'], title = "Visibilty of day4 in log")
plt.show()

df.drop(['day0_MI_visibilty','day1_MI_visibilty','day2_MI_visibilty','day3_MI_visibilty','day4_MI_visibilty'], axis=1, inplace=True)

##########################################
histogram_plot(df['day0_IN_tot_rain'], title = "Total rain of day0")
plt.show()
df['log_day0_IN_tot_rain'] = np.log(1 + df['day0_IN_tot_rain'])
histogram_plot(df['log_day0_IN_tot_rain'], title = "Total rain of day0 in log")
plt.show()


histogram_plot(df['day1_IN_tot_rain'], title = "Total rain of day1")
plt.show()
df['log_day1_IN_tot_rain'] = np.log(1 + df['day1_IN_tot_rain'])
histogram_plot(df['log_day1_IN_tot_rain'], title = "Total rain of day1 in log")
plt.show()


histogram_plot(df['day2_IN_tot_rain'], title = "Total rain of day2")
plt.show()
df['log_day2_IN_tot_rain'] = np.log(1 + df['day2_IN_tot_rain'])
histogram_plot(df['log_day2_IN_tot_rain'], title = "Total rain of day2 in log")
plt.show()


histogram_plot(df['day3_IN_tot_rain'], title = "Total rain of day3")
plt.show()
df['log_day3_IN_tot_rain'] = np.log(1 + df['day3_IN_tot_rain'])
histogram_plot(df['log_day3_IN_tot_rain'], title = "Total rain of day3 in log")
plt.show()


histogram_plot(df['day4_IN_tot_rain'], title = "Total rain of day4")
plt.show()
df['log_day4_IN_tot_rain'] = np.log(1 + df['day4_IN_tot_rain'])
histogram_plot(df['log_day4_IN_tot_rain'], title = "Total rain of day4 in log")
plt.show()

df.drop(['day0_IN_tot_rain','day2_IN_tot_rain','day2_IN_tot_rain','day3_IN_tot_rain',
         'day4_IN_tot_rain'], axis=1, inplace=True)


df = pd.get_dummies(df, prefix=[ 'Month','FireSeason','PrepLevel'], 
                    columns=['Month','FireSeason','PrepLevel'])

df.to_csv(r'C:\Users\toshiba\Desktop\fire forest\Data1\Feature_engineered.csv',index=False)




