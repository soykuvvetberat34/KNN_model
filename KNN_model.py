import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore",message='.*Could not find the number of physical cores.*')

#data preprocessing
datas=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\Makine Öğrenmesi\\KNN\\Hitters.csv")
datas=datas.dropna()
dms=pd.get_dummies(datas[["League","Division","NewLeague"]])
y=datas["Salary"]
x_=datas.drop(["Salary","Division","League","NewLeague"],axis=1).astype("float64")
x=pd.concat([x_,dms],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

KNN=KNeighborsRegressor()
KNN.fit(x_train,y_train)
predict=KNN.predict(x_test)
RMSE=np.sqrt(mean_squared_error(y_test,predict))
print(RMSE)

#MODEL TUNING

#1 ile 10 arasında tüm komşu sayıları ile modeli oluşturup
#oluşan rmse değerlerini inceleyeceğiz
RMSE_list=[]
for k in range(10):
    k=k+1
    knn_model=KNeighborsRegressor(n_neighbors=k)
    knn_model.fit(x_train,y_train)
    y_pred=knn_model.predict(x_test)
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    RMSE_list.append(rmse)
    print(f"k değeri {k} için rmse değeri {rmse}")

#k 8 için rmse nin en düşük değer olduğunu bulduk uygun olarak
#şimdi gridsearchcv ile bu değere ulaşmaya çalışalım

#bir dictionary oluşturup buraya oluşturdugumuz komşu sayıolarını koyalım
knn_params={
    "n_neighbors":np.arange(1,30,1)#1 ile 30 arasını birer birer ayır
    #toplam 29 tane komşu değeri
}
knn=KNeighborsRegressor()
knn_crossvalidation_model=GridSearchCV(knn,knn_params,cv=10)
#GridSearchCV(model,model_komşu_parametreleri,kaç katlı olacağı)
#modeli eğitelim
knn_crossvalidation_model.fit(x_train,y_train)
#knn_crossvalidation_model içerisinden gridsearchcv ile bulunmuş en iyi parametreyi alalım
best_parameter=knn_crossvalidation_model.best_params_
model_knn_tuned=KNeighborsRegressor(n_neighbors=best_parameter["n_neighbors"])
model_knn_tuned.fit(x_train,y_train)
predict_knn_tuned=model_knn_tuned.predict(x_test)
rmse_tuned=np.sqrt(mean_squared_error(y_test,predict_knn_tuned))
print(f"knn tuned rmse value: {rmse_tuned}")




