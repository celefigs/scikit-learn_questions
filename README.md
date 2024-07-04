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


## 8) Example machine learning workflow
The first 6 steps -> Pandas, the subsequent steps using scikit-learn and matplotlib

* 1. Read the data as a Pandas DataFrame and display the first and last few rows.
* 2. Display the dataset dimensions (rows and columns).
* 3. Display the data types of the columns and summarize how many columns are categorical and numerical types.
* 4. Handle missing data.
* 5. Perform Exploratory Data Analysis. Use Pandas `groupby` function (on categorical variables) together with `aggregate` function, as well as creating plots to explore data.
* 6. Assign independent variables to the 'X' variable while assigning the dependent variable to the 'y' variable.
* 7. Perform data splitting. Remember the random seed number.
* 8. Use the training set to build a machine learning model using the Random Forest algorithm.
* 9. Perform hyperparameter tuning coupled with cross-validation via the use of the GridSearchCV() function.
* 10. Apply the trained model to make predictions on the test set via the `predict()` function.
* 11. Explain the obtained model performance metrics as a summary table with the help of Pandas or display a visualization with Matplotlib.
* 12. Explain important features as identified by the Random Forest Model.





# Paso 4: Manejar datos faltantes si los hay
# Supongamos que queremos llenar los datos faltantes con la media de cada columna numérica
df.fillna(df.mean(), inplace=True)

# Paso 5: Realizar Análisis Exploratorio de Datos (EDA)
# Por ejemplo, utilizando la función groupby y agregación para analizar datos categóricos
# y creando visualizaciones usando Matplotlib
import matplotlib.pyplot as plt

# Ejemplo de EDA
print("\nEjemplo de Análisis Exploratorio de Datos:")
agg_data = df.groupby('categoria').aggregate({'columna_numerica': 'mean'})
print(agg_data)

plt.figure(figsize=(10, 6))
plt.bar(agg_data.index, agg_data['columna_numerica'])
plt.title('Media de columna_numerica por categoría')
plt.xlabel('Categoría')
plt.ylabel('Media de columna_numerica')
plt.xticks(rotation=45)
plt.show()

# Paso 6: Asignar variables independientes (X) y dependiente (y)
X = df.drop('variable_dependiente', axis=1)
y = df['variable_dependiente']

# Paso 7: Dividir los datos en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 8: Construir un modelo de aprendizaje automático usando Random Forest
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Paso 9: Ajuste de hiperparámetros con validación cruzada usando GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Paso 10: Aplicar el modelo entrenado para hacer predicciones sobre el conjunto de prueba
y_pred = grid_search.predict(X_test)

# Paso 11: Explicar las métricas de rendimiento del modelo obtenido
from sklearn.metrics import classification_report

print("\nInforme de clasificación:")
print(classification_report(y_test, y_pred))

# Paso 12: Explicar las características importantes identificadas por el modelo de Random Forest
importances = grid_search.best_estimator_.feature_importances_
feature_names = X.columns
feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importances = feature_importances.sort_values('importance', ascending=False)

print("\nCaracterísticas más importantes:")
print(feature_importances.head())

# Visualización de importancia de características (opcional)
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['feature'][:10], feature_importances['importance'][:10])
plt.xlabel('Importancia')
plt.title('Importancia de las características más importantes')
plt.gca().invert_yaxis()
plt.show()
