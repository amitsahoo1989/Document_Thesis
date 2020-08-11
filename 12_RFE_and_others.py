import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold

from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel


#######################################################################################


def normalize(df):
    
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def ml_model(X,y):
    
    X=normalize(X)
    #y=n_df['individual_total_acres']
    #X=n_df.drop(['individual_total_acres'], axis=1)
    #print((y))
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)
    models = {}
    models.update({'LogR': LogisticRegression(solver='saga',penalty='l1',max_iter=1000)})
    #models.update({'Bag_CL': BaggingClassifier()})
    #models.update({'RandomForest': RandomForestClassifier(n_estimators=500, criterion='entropy',)})
    #models.update({'ExtraTClf': ExtraTreesClassifier()})
    #models.update({'KNN': KNeighborsClassifier()})
    #models.update({'DT': DecisionTreeClassifier()})
    #models.update({'SVM': SVC()})
    #models.update({'XG':XGBClassifier()})
    #models.update({'ABC':AdaBoostClassifier()})
    #models.update({'GBC':GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)})
    
    #print(models)

    for i in models:
        model=models[i].fit(x_train,y_train)
        y_pred = model.predict(x_test)
        results = confusion_matrix(y_test, y_pred)
        print(i)
        print('===================================')
        print(results)
        print('Accuracy Score :',accuracy_score(y_test, y_pred) )
        #print('Report : ')
        #print(classification_report(y_test, y_pred))
    
    return True

file=r'C:\Users\toshiba\Desktop\fire forest\Data1\Feature_engineered.csv'
df=pd.read_csv(file)
df.shape
df['Fire_Severity'] = df['Fire_Severity'].astype('category',copy=False)
y=df['Fire_Severity']

####################Pearson correlation R##########################

sns.heatmap(df.corr())

######Recursive Feature Elimination#########################################

correlated_features = []
correlation_matrix = df.drop('Fire_Severity', axis=1).corr()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.append(colname)

#print(correlated_features) 
df.shape
df=df.drop(correlated_features,axis=1)
#print(df.columns)


df_corre=df.drop('Fire_Severity',axis=1)
df.shape
print('Models through Correlation Coefficient ->')

model=ml_model(df_corre,y)##accuracy of correlatio matrix

X = df.drop('Fire_Severity', axis=1)
target = df['Fire_Severity']


rfc = RandomForestClassifier(random_state=123)
rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='accuracy')
rfecv.fit(X, target)

print('Optimal number of features: {}'.format(rfecv.n_features_))

plt.figure(figsize=(16, 9))
plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)
plt.show()


#print(np.where(rfecv.support_ == False)[0])
X.drop(X.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)

df.shape
#print(X.columns.values.tolist())
df_rfe=df[X.columns.values.tolist()]
df_rfe.shape
print('Models through Recursive Feature Eelemination ->')

model=ml_model(df_rfe,y)##accuracy of correlatio matrix


dset = pd.DataFrame()
dset['attr'] = X.columns
dset['importance'] = rfecv.estimator_.feature_importances_

dset = dset.sort_values(by='importance', ascending=False)


plt.figure(figsize=(16, 14))
plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Importance', fontsize=14, labelpad=20)
plt.show()

##############RandomForest to select features based on feature importance.######

file=r'C:\Users\toshiba\Desktop\fire forest\Data1\Feature_engineered.csv'
df=pd.read_csv(file)

df['Fire_Severity'] = df['Fire_Severity'].astype('category',copy=False)

print('Models through random forest classifer ->')

y=df['Fire_Severity']
X=df.drop(['Fire_Severity'], axis=1)

rf = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=X.shape[1])
rf.fit(X, y)

embeded_rf_support = rf.get_support()
embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
print(str(len(embeded_rf_feature)), 'selected features by random forest feature selection')
#print("Selected features are -> ",embeded_rf_feature)
#model_run(df[embeded_rf_feature],y)
#print(embeded_rf_feature)

df_x=df[embeded_rf_feature]

model=ml_model(df_x,y)





###################Feature selection by boosting regressor######


from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier

print('Models through gradient boosting ->')
lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
            reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)

embeded_lgb_selector = SelectFromModel(lgbc, max_features=X.shape[1])
#print(y.dtypes)
embeded_lgb_selector.fit(X, y)

embeded_lgb_support = embeded_lgb_selector.get_support()
embeded_lgb_feature = X.loc[:,embeded_lgb_support].columns.tolist()
print(str(len(embeded_lgb_feature)), 'selected features')
#model_run(df[embeded_lgb_feature],y)
#print(embeded_lgb_feature)

df_x=df[embeded_lgb_feature]

model=ml_model(df_x,y)


##########################################################################################

boruta_columns=['individual_fire_num','day0_F_min','day0_F_mean',                
'day0_F_max',                 
'day0_F_Mean_dew_point',      
'day1_F_min',                 
'day1_F_mean',                
'day1_F_max',                 
'day1_IN_Sea_level',          
'day1_F_Mean_dew_point',      
'day1_IN_tot_rain',           
'day2_F_min',                 
'day2_F_mean',                
'day2_F_max',                 
'day2_IN_Sea_level',          
'day2_F_Mean_dew_point',      
'day2_MPH_max_wind',          
'day3_F_min',                 
'day3_F_mean',                
'day3_F_max',                 
'day3_IN_Sea_level',          
'day3_F_Mean_dew_point',      
'day3_MPH_mean_wind_speed',   
'day3_MPH_max_sustained_wind',
'day3_MPH_max_wind',          
'day4_F_min',                 
'day4_F_mean',                
'day4_F_max',                 
'day4_IN_Sea_level',          
'day4_F_Mean_dew_point',      
'day4_MPH_mean_wind_speed',   
'log_day2_MI_visibilty',      
'log_day3_MI_visibilty',      
'log_day4_MI_visibilty',      
'log_day0_IN_tot_rain',      
'log_day2_IN_tot_rain',      
'log_day1_IN_tot_rain',      
'log_day3_IN_tot_rain',     
'log_day4_IN_tot_rain',     
'Month_4',    
'Month_5',                  
'Month_6',                    
'Month_9',                    
'FireSeason_1994',            
'FireSeason_2002',            
'FireSeason_2003',            
'FireSeason_2004',            
'FireSeason_2005',           
'FireSeason_2006',            
'FireSeason_2007',            
'FireSeason_2008',            
'FireSeason_2009',            
'FireSeason_2010',           
'FireSeason_2011',            
'FireSeason_2012',            
'FireSeason_2014',            
'PrepLevel_1',                
'PrepLevel_2',                
'PrepLevel_3',                
'PrepLevel_4',                
'PrepLevel_5']

#print(len(boruta_columns))

df_bourta=df[boruta_columns]

#print(df_bourta.shape)
model=ml_model(df_bourta,y)

df_bourta['Fire_Severity']=y


df_bourta.to_csv(r'C:\Users\toshiba\Desktop\fire forest\Data1\feature_selection.csv',index=False)
