# scikit-learn_questions from "How to Master Scikit-learn for Data Science"
Chanin Nantasenamat

1) Why I want to create an artificial dataset?
2) what does feature scaling means and when should I use it? Example? por ejemlo, si tenemos alturas, pesos de personas, esas variabkles van a tener escalas muy diferentes y no es bueno para que el modelo aprenda, mejor adaptarlas para que vayan entre 0 y 1
3) "Creating a workflow using pipeline"? es un modo de trabajar, pero no necesariamente se debe utilizar.
4) feature scaling -> se usa para normalizar el rango de las variables independientes
5) Core steps: 

#### #0. Import 
____________________from scikit.modulename import EstimatorName()      
#### #1. Instantiate
____________________model = EstimatorName()                            
#### #2. Fit 
____________________model.fit(X_train, y_train)                        
#### #3. Predict 
____________________model.predict(X_test)                               
#### #4. Score 
____________________model.score(X_test, y_test)                        
