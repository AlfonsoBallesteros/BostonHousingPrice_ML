<h1 align="center">Welcome to BostonHousingPrice_ML 👋</h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-0.0.1-blue.svg?cacheSeconds=2592000" />
  <img alt="Python" src="https://img.shields.io/pypi/pyversions/pip" />
  <img alt="Language" src="https://img.shields.io/github/languages/top/AlfonsoBallesteros/BostonHousingPrice_ML" />
  <img alt="issues" src="https://img.shields.io/github/issues/AlfonsoBallesteros/BostonHousingPrice_ML" />
  <img alt="issues" src="https://img.shields.io/github/last-commit/AlfonsoBallesteros/BostonHousingPrice_ML" />
  <a href="https://github.com/AlfonsoBallesteros/BostonHousingPrice_ML/blob/master/LICENSE" target="_blank">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
  <a href="https://twitter.com/alfonsoballest4" target="_blank">
    <img alt="Twitter: alfonsoballest4" src="https://img.shields.io/twitter/follow/alfonsoballest4.svg?style=social" />
  </a>
</p>

# BostonHousingPrice_ML

  We will build an information system with regression machine learning that uses the 
  Housing-price-data dataset

# Problem

    For this project we will build an information system to predict housing prices in the city of
    Boston with the Boston House Price data set. 

# Objetive

      We will build an information system to predict the price of houses in the city of bostson. 
      We will develop the model with machine learning  and use the Boston House Price data set. 
      in order to find the price of the houses depending on the variables of the data set

      We will learn:

      * How to work through a regression predictive modeling problem end-to-end.
      * How to use data transforms to improve model performance.
      * How to use algorithm tuning to improve model performance.

      *How to use ensemble methods and tuning of ensemble methods to improve model performance.

## Outbuildings

    1. Numpy
    2. Matplotlib
    3. Pandas
    4. scikit-learn
      4.1. LinearRegression
      4.2. Lasso
      4.3. ElasticNet
      4.4. KNeighborsRegressor
      4.5. DecisionTreeRegressor
      4.6. Support Vector Machine
      4.7. Performace:
        4.7.1. RandomForestRegressor
        4.7.2. GradientBoostingRegressor
        4.7.3. ExtraTreeRegressor
        4.7.4. AdaBoostRegressor
      4.8. Metrics
        4.8.1. MeanSquaredError
    5. DataSet: Boston Housing Data (housing.csv)
    6. Backend: Flask
    7. Frontend: Vanilla Javascript

# Results

Metric | Result
-- | --
MeanSquaredError | 11.8752520792

    We can see that the estimated mean squared error is 11.8, close to our estimate of -9.3.

## Install

```sh
git clone https://github.com/AlfonsoBallesteros/BostonHousingPrice_ML.git
```

## Usage

```sh
python app.py
```
## Architecture Proyect
```sh
.
├── app.py
├── Backend
├── Dataset
│   └── housing.csv
├── Frontend
│   ├── index.html
│   └── script.js
├── Models
│   ├── housing_svc.model
│   └── scaler.scaler
├── README.md
├── readme.txt
└── Training
    ├── Regression_ML.ipynb
    └── Regression_ML.py
```

## Architecture API
```json
{
    "CRIM":  0.00632,          
    "ZN":  18.0, 
    "INDUS":  2.31,
    "CHAS":  0,
    "NOX": 0.538,
    "RM": 6.575,
    "AGE": 65.2,
    "DIS": 4.0900,
    "RAD": 1,
    "TAX": 296.0,
    "PTRATIO": 15.3,
    "B": 396.90,
    "LSTAT":  4.98
}
```
## Architecture Data

    1. CRIM: per capita crime rate by town
    2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
    3. INDUS: proportion of non-retail business acres per town
    4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    5. NOX: nitric oxides concentration (parts per 10 million)
    6. RM: average number of rooms per dwelling
    7. AGE: proportion of owner-occupied units built prior to 1940
    8. DIS: weighted distances to five Boston employment centers
    9. RAD: index of accessibility to radial highways
    10. TAX: full-value property-tax rate per $10,000
    11. PTRATIO: pupil-teacher ratio by town
    12. B: 1000(Bk − 0.63)2 where Bk is the proportion of blacks by town
    13. LSTAT: % lower status of the population
    14. MEDV: Median value of owner-occupied homes in $1000s

# Conclusion and Recommendations

    This project was built an information system capable of predicting the price of houses 
    in the city of Boston. We train models with different algorithms and optimization
    methods to obtain a more optimal result.

    good hacking!

# Plantilla de proyecto de Python

## 1. Prepare el problema
    a) Cargar bibliotecas
    b) Cargar conjunto de datos

## 2. Resumir datos
    a) Estadísticas descriptivas
    b) Visualizaciones de datos

## 3. Preparar datos
    a) Limpieza de datos
    b) Selección de características
    c) Transformaciones de datos

## 4. Evaluar algoritmos
    a) Conjunto de datos de validación dividida
    b) Opciones de prueba y métrica de evaluación
    c) Algoritmos de verificación puntual
    d) Comparar algoritmos

## 5. Mejora la precisión
    a) Ajuste del algoritmo
    b) Conjuntos

## 6. Finalizar modelo
    a) Predicciones sobre el conjunto de datos de validación
    b) Crear modelo independiente en todo el conjunto de datos de entrenamiento
    c) Guardar modelo para uso posterior

## Author

👤 **Alfonso Ballesteros**

* Twitter: [@alfonsoballest4](https://twitter.com/alfonsoballest4)
* Github: [@AlfonsoBallesteros](https://github.com/AlfonsoBallesteros)

## 🤝 Contributing

Contributions, issues and feature requests are welcome!<br />Feel free to check [issues page](https://github.com/AlfonsoBallesteros/BostonHousingPrice_ML/issues).

## Show your support

Give a ⭐️ if this project helped you!

## 📝 License

Copyright © 2019 [Alfonso Ballesteros](https://github.com/AlfonsoBallesteros).<br />
This project is [MIT](https://github.com/AlfonsoBallesteros/BostonHousingPrice_ML/blob/master/LICENSE) licensed.

## Expressions of Gratitude 🎁

* Tell others about this project 📢
* Invite a beer 🍺 to someone on the team.
* You give the thanks publicly 🤓.