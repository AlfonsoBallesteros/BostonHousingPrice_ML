#!/usr/bin/env python
# coding: utf-8

# In[1]:


#(Boston house price data).
#Boston House Price. Cada registro en la base de datos describe un suburbio o ciudad de Boston. 
#Los datos fueron extraídos del Boston Standard Metropolitan Área estadística (SMSA) en 1970


# In[2]:


#1.CRIM: tasa de criminalidad per cápita por ciudad
#2. ZN: proporción de tierra residencial dividida en zonas para lotes de más de 25,000 pies cuadrados.
#3. INDUS: proporción de acres de negocios no minoristas por ciudad
#4. CHAS: variable ficticia Charles River (= 1 si el tramo limita con el río; 0 en caso contrario)
#5. NOX: concentración de óxidos nítricos (partes por 10 millones)
#6.RM: número promedio de habitaciones por vivienda
#7. EDAD: proporción de unidades ocupadas por el propietario construidas antes de 1940
#8. DIS: distancias ponderadas a cinco centros de empleo de Boston
#9. RAD: índice de accesibilidad a las autopistas radiales.
#10. IMPUESTO: tasa impositiva sobre el valor total de la propiedad por $ 10,000
#11. PTRATIO: relación alumno-profesor por localidad
#12. B: 1000 (Bk - 0.63) 2 donde Bk es la proporción de negros por ciudad
#13. LSTAT:% menor estado de la población
#14. MEDV: valor medio de las viviendas ocupadas por sus propietarios en $ 1000
#Podemos ver que los atributos de entrada tienen una mezcla de unidades.


# In[2]:


# Load libraries
import numpy
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error


# In[3]:


# Load dataset
filename = '../Dataset/housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
'B', 'LSTAT', 'MEDV']
dataset = read_csv(filename, delim_whitespace=True, names=names)


# In[3]:


# shape
print(dataset.shape)


# In[7]:


# types
print(dataset.dtypes)


# In[8]:


# head
#Imprima las primeras filas del conjunto de datos.
print(dataset.head(20))


# In[9]:


#Imprima las descripciones estadísticas del conjunto de datos.
# descriptions
set_option('precision', 1)
print(dataset.describe())


# In[10]:


#Imprime las correlaciones entre los atributos.
# correlation
set_option('precision', 2)
print(dataset.corr(method='pearson'))


# In[11]:


#Data Visualizations


# In[13]:


#Unimodal Data Visualizations
#Visualice el conjunto de datos usando gráficos de histograma
# histograms
dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
pyplot.show()


# In[14]:


#Visualice el conjunto de datos utilizando gráficos de densidad.
# density
dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=False, legend=False,
fontsize=1)
pyplot.show()


# In[15]:


#Visualice el conjunto de datos usando diagramas de caja y bigotes.
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False,
fontsize=8)
pyplot.show()


# In[16]:


#Multimodal Data Visualizations
#Visualize the dataset using scatter plots.
# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()


# In[17]:


#Visualice las correlaciones entre atributos.
# correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = numpy.arange(0,14,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()


# In[8]:


#Validation Dataset
#Separe los datos en un conjunto de datos de capacitación y validación.
# Split-out validation dataset
array = dataset.values
X = array[:,0:13]
Y = array[:,13]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,
test_size=validation_size, random_state=seed)


# In[19]:


#Evaluate Algorithms: Baseline
# Test options and evaluation metric
#Configurar el arnés de prueba de evaluación de algoritmo.
num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'


# In[20]:


# Spot-Check Algorithms
#Cree la lista de algoritmos para evaluar.
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))


# In[22]:


#Evaluar la lista de algoritmos.
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[23]:


# Compare Algorithms
#Visualice las diferencias en el rendimiento del algoritmo.
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# In[24]:


#Evaluate Algorithms: Standardization
# Standardize the dataset
#Evaluar algoritmos en un conjunto de datos estandarizado.
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',
LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO',
Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN',
ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',
KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',
DecisionTreeRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[25]:


#Visualice las diferencias en el rendimiento del algoritmo en el conjunto de datos estandarizado.
# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# In[26]:


#Improve Results With Tuning
#Ajuste los parámetros del algoritmo KNN en el conjunto de datos estandarizado.
# KNN Algorithm tuning
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
k_values = numpy.array([1,3,5,7,9,11,13,15,17,19,21])
param_grid = dict(n_neighbors=k_values)
model = KNeighborsRegressor()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)


# In[27]:


#Salida de impresión al ajustar el algoritmo KNN.
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[28]:


#Ensemble Methods
# ensembles
ensembles = []
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB',
AdaBoostRegressor())])))
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM',
GradientBoostingRegressor())])))
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF',
RandomForestRegressor())])))
ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET',
ExtraTreesRegressor())])))


# In[30]:


#Evaluar algoritmos de conjunto en el conjunto de datos estandarizado.
results = []
names = []
for name, model in ensembles:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[31]:


# Compare Algorithms
#Visualice las diferencias en el rendimiento del algoritmo de conjunto en estandarizado
fig = pyplot.figure()
fig.suptitle('Scaled Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# In[32]:


#Tune Ensemble Methods
# Tune scaled GBM
#Ajuste GBM en el conjunto de datos escalado.
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators=numpy.array([50,100,150,200,250,300,350,400]))
model = GradientBoostingRegressor(random_state=seed)
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)


# In[34]:


#Rendimiento de salida de GBM sintonizado en conjunto de datos escalado.
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[9]:


#Finalize Model
# prepare the model
#Construir el modelo finalizado.
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = GradientBoostingRegressor(random_state=seed, n_estimators=400)
model.fit(rescaledX, Y_train)


# In[10]:


# transform the validation dataset
#Resultado de la evaluación del modelo finalizado.
import numpy as np
rescaledValidationX = scaler.transform(X_validation)
print(X_validation[0:1])
print(rescaledValidationX[0:1])
predictions = model.predict(rescaledValidationX)
print(np.argmax(predictions))
print(mean_squared_error(Y_validation, predictions))


# In[11]:


import pandas as pd
#creo un dataframe con todos los datos
df = pd.DataFrame([[0.00632, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.0900, 1, 296.0, 15.3, 396.90, 4.98]])
#transformo los datos
res = scaler.transform(df)
#hago la prediccion
pre = model.predict(res)
p = float(pre)
print (format(p, '.3f'))


# In[157]:


from sklearn.externals import joblib
#Save Model
joblib.dump(model,'../Models/housing_svc.model') 


# In[12]:


#Load the model
house = joblib.load('../Models/housing_svc.model') 
#creo un dataframe con todos los datos
data = [[0.00632, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.0900, 1, 296.0, 15.3, 396.90, 4.98]]
#hago la prediccion
r = scaler.transform(data)
pro = house.predict(r)
o = float(pro)
print (format(o, '.3f'))


# In[161]:


# save the model to disk
filename = '../Models/scaler.scaler'
joblib.dump(scaler, filename)


# In[ ]:




