# scikit-learn_questions from "How to Master Scikit-learn for Data Science"
Chanin Nantasenamat

1) Why I want to create an artificial dataset?
2) what does feature scaling means and when should I use it? Example? por ejemlo, si tenemos alturas, pesos de personas, esas variables van a tener escalas muy diferentes y no es bueno para que el modelo aprenda, mejor adaptarlas para que vayan entre 0 y 1
3) "Creating a workflow using pipeline"? es un modo de trabajar, pero no necesariamente se debe utilizar.
4) feature scaling -> se usa para normalizar el rango de las variables independientes
5) Core steps code: 

- #0. Import 
- #1. Instantiate 
- #2. Fit 
- #3. Predict 
- #4. Score
-----  
- #0. from scikit.modulename import EstimatorName() 
- #1. model = EstimatorName()
- #2. model.fit(X_train, y_train)
- #3. model.predict(X_test)
- #4. model.score(X_test, y_test)  

6) Model interpretation: ¿qué características son las más importantes?
  rf.feature_importances --> produces an array of importance values, then can create a feature importance plot.
7) Hyperparameter* tuning:
hyperparameters* son configuraciones que se establecen de antemano en el proceso de aprendizaje de un a modelo, determinan como de bien aprende un modelo.
 1. Create an artificial dataset and perf data splitt.
 2. Perform the actual hyperparm tuning
 3. display the results form hyper tuning in a visual representation.

Se supone que este enfoque ayuda a encontrar la configuración óptima que maximiza el rendimiento del modelo en nuevos datos.

#### Artificial Dataset 
Cuando creamos un dataset artificial, estamos generando datos simulados que imitan características reales. Por ejemplo, podríamos crear un conjunto de datos con características como edad, ingresos y estado civil, pero estos datos no serían de personas reales, sino generados por computadora para propósitos de entrenamiento y prueba de modelos.

#### Uso de Hiperparámetros
Los hiperparámetros son configuraciones que ajustamos antes de entrenar un modelo de machine learning. Son como ajustes o configuraciones que podemos modificar para mejorar el rendimiento del modelo. Por ejemplo, en un algoritmo como el bosque aleatorio (Random Forest), los hiperparámetros podrían ser el número de árboles en el bosque, la profundidad máxima de los árboles o la cantidad mínima de muestras requeridas para dividir un nodo.

### Relación entre ambos
Cuando creamos un dataset artificial, podemos diseñarlo específicamente para probar diferentes configuraciones de hiperparámetros y ver cómo afectan al rendimiento del modelo. Por ejemplo, podríamos ajustar los hiperparámetros del bosque aleatorio y entrenarlo con nuestro dataset artificial para ver cuál combinación produce mejores resultados (como mayor precisión o menor error).

En resumen, creamos datasets artificiales para entrenar modelos de machine learning y ajustamos los hiperparámetros para optimizar el rendimiento de esos modelos.

## 8) example machine learning workflow 
the 1rst 6 stesps -> Pandas, the subsequent steps using scikit-learn and matplotlib

1- Read the data as aPandas DataFrame and display the 1rst and last few rowws
2- Display the dataset dimension (rows and column)
3- Display the data types of the columnes and summerize how many columns are categorical and numerial types
4- 
