import time
from bs4 import BeautifulSoup  
from urllib.request import Request, urlopen
import csv,os
import pandas as pd






def getdetails(URL):
    req = Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    soup = BeautifulSoup(webpage, 'html.parser')
    
    table = soup.find('table',attrs={"class":"weatherhistory_results"})
    to_add=[]
    
    try:
        min_temp = table.find('tr',attrs={"class":"weatherhistory_results_datavalue temp_mn"})
        if('class="nullvalue">No data.' in str(min_temp)):
            ##print("No data")
            to_add.append('')
        else:
            min_temp=min_temp.find('span',attrs={"class":"value"})
            min_temp=(str(min_temp.contents).strip("['']"))
            to_add.append(min_temp)
            ##print(to_add)
        
        mean_temp = table.find('tr',attrs={"class":"weatherhistory_results_datavalue temp"})
        if('class="nullvalue">No data.' in str(mean_temp)):
            ##print("No data")
            to_add.append('')
        else:
            mean_temp=mean_temp.find('span',attrs={"class":"value"})
            mean_temp=(str(mean_temp.contents).strip("['']"))
            to_add.append(mean_temp)
            ##print(to_add)
    
        max_temp = table.find('tr',attrs={"class":"weatherhistory_results_datavalue temp_mx"})
        if('class="nullvalue">No data.' in str(max_temp)):
            ##print("No data")
            to_add.append('')
        else:
            max_temp = max_temp.find('span',attrs={"class":"value"})
            max_temp=(str(max_temp.contents).strip("['']"))
            to_add.append(max_temp)
            ##print(to_add)
    
        sea_lvl_pres = table.find('tr',attrs={"class":"weatherhistory_results_datavalue slp"})
        if('class="nullvalue">No data.' in str(sea_lvl_pres)):
            ##print("No data")
            to_add.append('')
        else:
            sea_lvl_pres=sea_lvl_pres.find('span',attrs={"class":"value"})
            sea_lvl_pres=(str(sea_lvl_pres.contents).strip("['']"))
            to_add.append(sea_lvl_pres)
            ##print(to_add)
        
    
        dew_pnt = table.find('tr',attrs={"class":"weatherhistory_results_datavalue dewp"})
        if ('class="nullvalue">No data.' in str(dew_pnt)):
            ##print('No data')
            to_add.append('')
        else:
            dew_pnt= dew_pnt.find('span',attrs={"class":"value"})
            dew_pnt=(str(dew_pnt.contents).strip("['']"))
            to_add.append(dew_pnt)
            #print(to_add)
        
        prcp = table.find('tr',attrs={"class":"weatherhistory_results_datavalue prcp"})
        if ('class="nullvalue">No data.' in str(prcp)):
            ##print('No data')
            to_add.append('')
    
        else:
            prcp= prcp.find('span',attrs={"class":"value"})
            prcp=(str(prcp.contents).strip("['']"))
            to_add.append(prcp)
            #print(to_add)
        
        visib = table.find('tr',attrs={"class":"weatherhistory_results_datavalue visib"})
        if ('class="nullvalue">No data.' in str(visib)):
            ##print('No data')
            to_add.append('')
        else:
            visib= visib.find('span',attrs={"class":"value"})
            visib=(str(visib.contents).strip("['']"))
            to_add.append(visib)
            #print(to_add)
        
        sndp = table.find('tr',attrs={"class":"weatherhistory_results_datavalue sndp"})
        if ('class="nullvalue">No data.' in str(sndp)):
            #print('No data')
            to_add.append('')
        else:
            sndp= sndp.find('span',attrs={"class":"value"})
            sndp=(str(sndp.contents).strip("['']"))
            to_add.append(sndp)
            #print(to_add)
        
        wdsp = table.find('tr',attrs={"class":"weatherhistory_results_datavalue wdsp"})
        if ('class="nullvalue">No data.' in str(wdsp)):
            #print('No data')
            to_add.append('')
        else:
            wdsp= wdsp.find('span',attrs={"class":"value"})
            wdsp=(str(wdsp.contents).strip("['']"))
            to_add.append(wdsp)
            #print(to_add)
        
        mxspd = table.find('tr',attrs={"class":"weatherhistory_results_datavalue mxspd"})
        if ('class="nullvalue">No data.' in str(mxspd)):
            #print('No data')
            to_add.append('')
        else:
            mxspd= mxspd.find('span',attrs={"class":"value"})
            mxspd=(str(mxspd.contents).strip("['']"))
            to_add.append(mxspd)
            #print(to_add)
        
        gust = table.find('tr',attrs={"class":"weatherhistory_results_datavalue gust"})
        if ('class="nullvalue">No data.' in str(gust)):
            #print('No data')
            to_add.append('')
        else:
            gust= gust.find('span',attrs={"class":"value"})
            gust=(str(gust.contents).strip("['']"))
            to_add.append(gust)
            #print(to_add)
        metro_values=to_add
        
    except:
        metro_values=[]
    
    return metro_values

file_name=r'C:\Users\toshiba\Desktop\fire forest\Data1\url.csv'

column_names=['URL','Remarks','F_min','F_mean','F_max','IN_Sea_level',
              'F_Mean_dew_point','IN_tot_rain','MI_visibilty','Snow_depth','MPH_mean_wind_speed',
              'MPH_max_sustained_wind','MPH_max_wind']

df = pd.DataFrame(columns=column_names)


with open(file_name,'r') as file1:
    reader=csv.DictReader(file1)
    counter=0
    
    ##for i in reader:
    for i in reader:
        counter=counter+1
        print("counter is at {}".format(counter))
        if (counter%20==0):
            print("counter is at rest")
            time.sleep(30)
        
            
            
        url=i['Day-0']
        metro_values=getdetails(url)
        #print(metro_values)
        
        
        if (len(metro_values)==11):
            metro_values.insert(0,url) 
            metro_values.insert(1,'')
            ##print(metro_values)
            df.loc[len(df)]=metro_values
            
        else:
            remark=metro_values
            to_be_inserted=[url,remark,'', '', '', '', '', '', '', '', '', '','']
            df.loc[len(df)]=to_be_inserted
            
        if (counter%50==0):
            file_path= r'C:\Users\toshiba\Desktop\fire forest\Images'
            filename="export_dataframe_day0"+str(counter)+'.csv'
            complete_filename=os.path.join(file_path,filename)
            df.to_csv (complete_filename, index = True, header=True)
            
df.to_csv(r'C:\Users\toshiba\Desktop\fire forest\Data1\export_dataframe_day0.csv', index = False, header=True)