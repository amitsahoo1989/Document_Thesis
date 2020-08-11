import pandas as pd
import random

####################################################################
fire_data=r'C:\Users\toshiba\Desktop\fire forest\Data1\fire_weather_final.csv'
df2=pd.read_csv(fire_data)
#print(df2)
#print(df2.shape)

individual_fire_num=[]
inividual_area_burnt=[]
index_total_fires=df2.columns.values.tolist().index("TotalFires")
index_total_acres=df2.columns.values.tolist().index("TotalAcres")



for i,j in enumerate(df2.loc[:,'TotalFires']):
 
    if i==0:
        individual_fire_num.append(df2.iloc[0,index_total_fires])
        
    else:
        if df2.iloc[i,1]!=df2.iloc[i-1,1]:
            ##print(f'chnaged from {df2.iloc[i-1,1]} to {df2.iloc[i,1]}')
            individual_fire_num.append(df2.iloc[i,index_total_fires])
        else:
            individual_fire_num.append(j-df2.iloc[i-1,index_total_fires])
            
            
print(len(individual_fire_num))



i=0
j=0
for i,j in enumerate(df2.loc[:,'TotalAcres']):
 
    if i==0:
        inividual_area_burnt.append(df2.iloc[0,index_total_acres])
        
    else:
        if df2.iloc[i,1]!=df2.iloc[i-1,1]:
            #print(f'chnaged from {df2.iloc[i-1,1]} to {df2.iloc[i,1]}')
            inividual_area_burnt.append(df2.iloc[i,index_total_acres])
        else:
            inividual_area_burnt.append(j-df2.iloc[i-1,index_total_acres])
            
#print(individual_fire_num)
            
df2.insert(4, "individual_fire_num", individual_fire_num , True) 
df2.insert(6, "individual_total_acres", inividual_area_burnt , True) 

df3=df2

#############################################################################
##Removing the duplicacy in the dataset
#print('shape of dataset size',df3.shape)

count=0
for i in df3['individual_total_acres']:
    if(i<0):
        count=count+1
#print('number of instances where burnt area is less than zero ',count)

my_list=[]
duplicate_dic={}
for j,i in enumerate(df3['SitReportDate']):
    if i in my_list:
        duplicate_dic.update({j:i})
    else:
        my_list.append(i)
    
#print(list(duplicate_dic.keys()))
#print(duplicate_dic)

df3=df3.drop(df3.index[list(duplicate_dic.keys())])
#print('final dataset size',df3.shape)

###############################################################################
##removing the negetives
df=df3

##df['D'] = df['individual_fire_num'] 

individual_fire_num=[]
individual_total_acres=[]

for i in df['individual_fire_num']:
    if i >=0:
        individual_fire_num.append(i)

for i in df['individual_total_acres']:
    if i>=0:
        individual_total_acres.append(i)

##(sum(individual_fire_num))
mean_individual_fire_num=sum(individual_fire_num)/len(individual_fire_num)

mean_individual_total_acres=sum(individual_total_acres)/len(individual_total_acres)

##print(mean_individual_total_acres)

######################################################################################
lsit_individual_fire_num=[i for i in df['individual_fire_num'] ]

#print(len(lsit_individual_fire_num))

for n, i in enumerate(lsit_individual_fire_num):
    if i<0:
        lsit_individual_fire_num[n] = random.randint(3,7)
        

#print(lsit_individual_fire_num)

df['individual_fire_num']=lsit_individual_fire_num
#####################################################################################
individual_total_acres=[i for i in df['individual_total_acres'] ]

#print(len(individual_total_acres))

for n, i in enumerate(individual_total_acres):
    if i<0:
        individual_total_acres[n] = random.randint(14123,14323)
        

#print(individual_total_acres)

df['individual_total_acres']=individual_total_acres
#####################################################################################
#####################Deting the record with no weather data and extarctning month####
#####################################################################################

#print (df.isnull().sum())

    
index_to_delete=df[df['day0_F_min'].isnull()].index.tolist()
#print(index_to_delete)
df=df.drop(index_to_delete[0]) ##Deleting the record with no weather data

#print (df.isnull().sum())
#print(df.shape)
###############################
max_wind=df['day0_MPH_max_wind']
#print(len(max_wind))
maxwind=['' if x==999 else x for x in max_wind]
#print(len(maxwind))
df['day0_MPH_max_wind']=maxwind
#################################
max_wind=df['day1_MPH_max_wind']
#print(len(max_wind))
maxwind=['' if x==999 else x for x in max_wind]
#print(len(maxwind))
df['day1_MPH_max_wind']=maxwind
#####################################
#################################
max_wind=df['day2_MPH_max_wind']
#print(len(max_wind))
maxwind=['' if x==999 else x for x in max_wind]
#print(len(maxwind))
df['day2_MPH_max_wind']=maxwind
################################
#################################
max_wind=df['day3_MPH_max_wind']
#print(len(max_wind))
maxwind=['' if x==999 else x for x in max_wind]
#print(len(maxwind))
df['day3_MPH_max_wind']=maxwind
###############################
#################################
max_wind=df['day4_MPH_max_wind']
#print(len(max_wind))
maxwind=['' if x==999 else x for x in max_wind]
#print(len(maxwind))
df['day4_MPH_max_wind']=maxwind


df = df.drop(columns=['ID','TotalFires','TotalAcres'],axis=1)
df.to_csv(r'C:\Users\toshiba\Desktop\fire forest\Data1\Pre_processed.csv',index=False)

