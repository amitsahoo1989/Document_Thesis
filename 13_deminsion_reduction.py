import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sns


def normalize(df):
    
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def ml_model(X,y):
    
    X=normalize(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)
    models = {}
    models.update({'LogR': LogisticRegression(solver='saga',penalty='l1',max_iter=1000)})
    
    for i in models:
        model=models[i].fit(x_train,y_train)
        y_pred = model.predict(x_test)
        results = confusion_matrix(y_test, y_pred)
        
        print(i)
        print('===================================')
        print(results)
        print('Accuracy Score :',accuracy_score(y_test, y_pred) )
        print('Report : ')
        print(classification_report(y_test, y_pred))
        
        
        
        
    
    return True


file=r'C:\Users\toshiba\Desktop\fire forest\data1\feature_selection.csv'
df_boruta=pd.read_csv(file)

#print(df_boruta.columns)

df_boruta['Fire_Severity'] = df_boruta['Fire_Severity'].astype('category',copy=False)

y=df_boruta['Fire_Severity']
df_boruta=df_boruta.drop('Fire_Severity',axis=1)
    
###########################Feature engineering finishes here###########################
###PCA#####    
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df_boruta.values)
print(pca_result)

plt.plot(range(3), pca.explained_variance_ratio_)
plt.plot(range(3), np.cumsum(pca.explained_variance_ratio_))
plt.title("Component-wise and Cumulative Explained Variance")

df_pca=pd.DataFrame(pca_result,columns=['Comp1','Comp2','Comp3'])
model=ml_model(df_pca,y)



####FCA#####
from sklearn.decomposition import FastICA 
ICA = FastICA(n_components=3, random_state=12) 
ica_result=ICA.fit_transform(df_boruta.values)
print(ica_result)

plt.figure(figsize=(12,8))
plt.title('ICA Components')
plt.scatter(ica_result[:,0], ica_result[:,1])
plt.scatter(ica_result[:,1], ica_result[:,2])
plt.scatter(ica_result[:,2], ica_result[:,0])
df_ica=pd.DataFrame(ica_result,columns=['Comp1','Comp2','Comp3'])
model=ml_model(df_ica,y)



########TSNE#########
from sklearn.manifold import TSNE
mylist=[]
for j in y.values.tolist():
    if j =='Small':
        mylist.append(0)
    
    else:
        mylist.append(1)
label1=np.array(mylist)

tsne = TSNE(n_components=3, n_iter=1000,n_iter_without_progress=5,perplexity=30.0,random_state=123).fit_transform(df_boruta.values)


plt.figure(figsize=(12,8))
plt.title('t-SNE components')
plt.scatter(tsne[:,0], tsne[:,1])
plt.scatter(tsne[:,1], tsne[:,2])
plt.scatter(tsne[:,2], tsne[:,0])

#print(matplotlib."__version__")
#from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')
x1 = np.array(tsne[:,0])
y1 = np.array(tsne[:,1])
z1 = np.array(tsne[:,2])

ax.scatter(x1,y1,z1, marker="s", c=label1 , s=40, cmap="jet")
plt.show()

df_tsne=pd.DataFrame(tsne,columns=['Comp1','Comp2','Comp3'])
model=ml_model(df_tsne,y)
    




from sklearn import manifold 
trans_data = manifold.Isomap(n_neighbors=100, 
                             n_components=3, 
                             n_jobs=-1).fit_transform(df_boruta.values)
plt.figure(figsize=(12,8))
plt.title('Decomposition using ISOMAP')
plt.scatter(trans_data[:,0], trans_data[:,1])
plt.scatter(trans_data[:,1], trans_data[:,2])
plt.scatter(trans_data[:,2], trans_data[:,0])
df_isomap=pd.DataFrame(trans_data,columns=['Comp1','Comp2','Comp3'])
print(df_isomap.shape)
model=ml_model(df_isomap,y)

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')
x1 = np.array(trans_data[:,0])
y1 = np.array(trans_data[:,1])
z1 = np.array(trans_data[:,2])

ax.scatter(x1,y1,z1, marker="s", c=label1 , s=40, cmap="jet")
plt.show()




from sklearn.decomposition import FactorAnalysis
FA = FactorAnalysis(n_components = 3).fit_transform(df_boruta.values)

plt.figure(figsize=(12,8))
plt.title('Factor Analysis Components')
plt.scatter(FA[:,0], FA[:,1])
plt.scatter(FA[:,1], FA[:,2])
plt.scatter(FA[:,2],FA[:,0])
df_factanalysis=pd.DataFrame(FA,columns=['Comp1','Comp2','Comp3'])
print(df_factanalysis.shape)
model=ml_model(df_factanalysis,y)

import umap
umap_data = umap.UMAP(n_neighbors=100, min_dist=0.3, 
                      n_components=3).fit_transform(df_boruta.values)
plt.figure(figsize=(12,8))
plt.title('Decomposition using UMAP')
plt.scatter(umap_data[:,0], umap_data[:,1])
plt.scatter(umap_data[:,1], umap_data[:,2])
plt.scatter(umap_data[:,2], umap_data[:,0])
df_umap=pd.DataFrame(umap_data,columns=['Comp1','Comp2','Comp1'])
print(df_umap.shape)
model=ml_model(df_umap,y)

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')
x1 = np.array(umap_data[:,0])
y1 = np.array(umap_data[:,1])
z1 = np.array(umap_data[:,2])

ax.scatter(x1,y1,z1, marker="s", c=label1 , s=40, cmap="jet")
plt.show()

##############################Dimesion reduction Finishes here###########################

##########################Model Building starts ffrom here###############################
print(df_tsne.shape,y.shape)
plt.hist(y)
plt.show()

sm = SMOTE(random_state=123)
X_smote,y_smote = sm.fit_sample(df_tsne, y)

plt.hist(y_smote)
plt.show()




df_tsne_smote=pd.DataFrame(X_smote,columns=['Comp1','Comp2','Comp3'])

    

x_train, x_test, y_train, y_test = train_test_split(df_tsne_smote, y_smote, test_size=0.20, random_state=123)



models = {}
#models.update({'LogR': LogisticRegression(penalty='l2',C=1.0,solver='newton-cg')})
#models.update({'Bag_CL': BaggingClassifier()})
models.update({'RandomForest': RandomForestClassifier(n_estimators=500, criterion='entropy',)})
#models.update({'ExtraTClf': ExtraTreesClassifier()})
#models.update({'KNN': KNeighborsClassifier(metric='euclidean', n_neighbors= 5, weights= 'distance')})
#models.update({'DT': DecisionTreeClassifier(max_leaf_nodes=99, min_samples_split=13, random_state=42)})
#models.update({'SVM': SVC()})
#models.update({'XG':XGBClassifier()})
#models.update({'ABC':AdaBoostClassifier()})
#models.update({'GBC':GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)})



def scaling_func(x_train,x_test):
    if i in ['LogR','KNN','SVM']:
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_train_transformed=scaler.transform(x_train)

        scaler.fit(x_test)
        x_test_transformed=scaler.transform(x_test)
    else:
        x_train_transformed=x_train
        x_test_transformed=x_test
    return x_train_transformed,x_test_transformed



##from sklearn.model_selection import GridSearchCV
for i in models:
    
    x_train_transformed,x_test_transformed=scaling_func(x_train,x_test)
    
    mdl=models[i]
    #print(mdl)
    #grid_values = {'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25]}
    #grid_clf_acc = GridSearchCV(mdl, param_grid = grid_values)
    mdl.fit(x_train_transformed,y_train)
    
    y_pred = mdl.predict(x_test_transformed)
    results = confusion_matrix(y_test, y_pred)
    print(i)
    print('===================================')
    print(results)
    print('Accuracy Score :',accuracy_score(y_test, y_pred) )
    print('Report : ')
    print(classification_report(y_test, y_pred))
    
    df_cm = pd.DataFrame(results, columns=np.unique(y_test), index = np.unique(y_test))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (10,7))
    sns.set(font_scale=1.4)#for label size
    sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size
    


y_ordinal1=pd.Series(y_smote)
y_ordinal1.shape

kf = KFold(n_splits=3,shuffle=False)
kf.split(df_tsne)  
accuracy_model = []
for train_index, test_index in kf.split(df_tsne):
    # Split train-test
    #print(train_index,test_index)
    
    
    x_train_cv, x_test_cv = df_tsne.iloc[train_index], df_tsne.iloc[test_index]
    scaler.fit(x_train_cv)
    x_train_transformed=scaler.transform(x_train_cv)
    #print(x_train.shape,x_test.shape)
    
    y_train_cv, y_test_cv = y_ordinal1.iloc[train_index], y_ordinal1.iloc[test_index]
    #print(y_train_cv)
    
    
        
    svc = SVC(kernel='rbf',degree=20)
    md = svc.fit(x_train_transformed, y_train_cv)
    
    scaler.fit(x_test_cv)
    x_test_transformed=scaler.transform(x_test_cv)
    
    y_pred= md.predict(x_test_transformed)
    #print(accuracy_score(y_test_cv, y_pred))
    accuracy_model.append(accuracy_score(y_test_cv, y_pred))
    
    
avg = sum(accuracy_model)/len(accuracy_model)
print("The average is ", round(avg,2))







print(df_tsne.shape,y.shape)
sm = SMOTE(random_state=123)
X_smote,y_smote = sm.fit_sample(df_tsne, y)
print(X_smote.shape,y_smote.shape)

from sklearn.model_selection import GridSearchCV
C = np.logspace(0, 5, num=20)
penalty = ['l1', 'l2', 'elasticnet', 'none']
solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
hyperparameters = dict(C=C, penalty=penalty, solver=solver)

logistic = LogisticRegression()
gridsearch = GridSearchCV(logistic, hyperparameters)
best_model = gridsearch.fit(X_smote, y_smote)

print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
print('Best Solver:', best_model.best_estimator_.get_params()['solver'])

print(best_model.best_estimator_.get_params())

################################################################


from sklearn.model_selection import GridSearchCV
grid_params={'n_neighbors':[3,5,7,11,15,19],
             'weights':['uniform','distance'],
             'metric':['euclidean','manhattan']}
gs=GridSearchCV(KNeighborsClassifier(), grid_params, verbose=1, cv=3, n_jobs=-1)
gs_results=gs.fit(X_smote,y_smote)

gs_results.best_score_
gs_results.best_params_






























        
    