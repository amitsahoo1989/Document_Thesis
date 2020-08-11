import csv
import pendulum 
num_of_past_days=5
def read_csv(file_csv):
    url_link='https://www.almanac.com/weather/history/AK/Akiachak/'
    name=r"C:\Users\toshiba\Desktop\fire forest\Data1\url.csv"
    
    with open(file_csv,'r') as file:
        reader= csv.DictReader(file)
        counter=0
        for i in reader:
            url_link_created=[]
            list_of_dates=[]
            
            year=i['SitReportDate'][0:4]
            month=i['SitReportDate'][4:6]
            day=i['SitReportDate'][6:8]
            
            for j in range(num_of_past_days):
                now = pendulum.datetime(int(year),int(month),int(day))
                yesterday = now.subtract(days=j)
                yesterday=str(yesterday)
                list_of_dates.append(yesterday[0:10])
            
            
            for k in list_of_dates:
                url_link_created.append(url_link+str(k))
            
            
           
            with open(name, 'a',newline='') as f:
                
                if (counter==0):
                    
                    writer = csv.DictWriter(f, fieldnames = ['Day-'+str(m) for m in range(num_of_past_days)])
                    writer.writeheader()
                    
                writer = csv.writer(f)
                writer.writerow(url_link_created)
            counter=counter+1
    return name
          

file_csv=r'C:\Users\toshiba\Desktop\fire forest\Data1\Alaska.csv'
name=read_csv(file_csv)
#print(name)
