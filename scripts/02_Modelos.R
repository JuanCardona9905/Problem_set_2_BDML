##################################################
###################### Data ######################
##################################################

#### 1. Cargar Paquetes ----
{
#se borra la memoria
rm(list = ls())
#se cargan los paquetes
library(pacman)
p_load(rio, # importación/exportación de datos
       tidyverse, # datos ordenados (ggplot y Tidyverse)
       skimr, # datos de resumen
       visdat, # visualización de datos faltantes
       corrplot, # gráficos de correlación
       stargazer, # tablas/salida a TEX.
       rvest, # web-scraping
       readxl,
       readr, # importar Excel
       writexl, # exportar Excel
       boot, # bootstrapping
       ggpubr, # extensiones de ggplot2
       WVPlots, # gráficos de variables ponderadas
       patchwork, # para combinar gráficos
       gridExtra, # para combinar gráficos
       ggplot2, # gráficos
       caret, # para evaluación de modelos predictivos
       glmnet, # para evaluación de modelos predictivos
       data.table, # para manipulación de datos
       MASS, # El paquete tiene a la funcion de LDA
       class, # El paquete tiene a la funcion de k-neighbours
       gmodels,
       tree,
       naniar) # missing

#se define la ruta de trabajo
ifelse(grepl("camilabeltran", getwd()),
       wd <- "/Users/camilabeltran/OneDrive/Educación/PEG - Uniandes/BDML/GitHub/problem_set/Problem_set_1",
       ifelse(grepl("Juan",getwd()),
              wd <- "C:/Users/Juan/Documents/Problem_set_2",
              ifelse(grepl("juanp.rodriguez",getwd()),
                     wd <- "C:/Users/juanp.rodriguez/Documents/GitHub/Problem_set_1",
                     ifelse(grepl("C:/Users/User",getwd()),
                            wd <- "C:/Users/User/OneDrive - Universidad de los andes/Big Data y Machine Learning/Problem_set_1/Problem_set_1",
                            ifelse(grepl("/Users/aleja/",getwd()),
                                   wd <- "/Users/aleja/Documents/Maestría Uniandes/Clases/Big Data y Machine Learning/Repositorios Git Hub/Problem_set_1)",
                                   wd <- "otro_directorio")))))

#Script: "01_web_scraping.R". Realiza el proceso de web scraping para conseguir los datos
setwd(paste0(wd,"/scripts"))
}

#### 2. Importar bases de datos ----
{
### Importar las bases de datos finales 
# Train

# Test
  
}


#### 3. Modelos ----

### Logistic Regression --

### Modelo 1 -
{
## Entrenamiento

# Modelo
Mod_1_LR <- glm(formula = Pobre ~., 
                data = train, 
                family = binomial)
summary(Mod_1_LR)

# Probabilidades in-sample
Mod_1_LR_prob <- predict(Mod_1_LR, type="response")
# Hacer un histograma de las probabilidades

# Prediccion in-sample
Mod_1_LR_pred <- ifelse(Mod_1_LR_prob>0.5, "Pobre", "No Pobre")

# Matriz de confusion
table(Mod_1_LR_pred, train$Pobre) #aca se puede ajustar el umbral
mean(Mod_1_LR_pred == train$Pobre)

# Probabilidades out-sample
Mod_1_LR_prob <- predict(Mod_1_LR, newdata = test, type="response")
# Hacer un histograma de las probabilidades

# Prediccion out-sample
Mod_1_LR_pred <- ifelse(Mod_1_LR_prob>0.5, "Pobre", "No Pobre")

}

### Linear Discriminant Analysis --

### Modelo 2 -
{
  ## Entrenamiento
  
  # Modelo
  Mod_2_LDA <- lda(formula = Pobre ~., 
                  data = train)
  Mod_2_LDA
  plot(Mod_2_LDA)

  # Prediccion in-sample
  Mod_2_LDA_pred <- predict(Mod_2_LDA, train)
  
  # Matriz de confusion
  table(Mod_2_LDA_pred, train$Pobre) #aca se puede ajustar el umbral
  mean(Mod_1_LR_pred == train$Pobre)
  
  # Prediccion out-sample
  Mod_2_LDA_pred <- predict(Mod_2_LDA, test)
  # Borrar Creo que toca con data.frame...
}

### K-nearest Neighbour Classification --

### Modelo 3 -
{
# Es necesario los datos en forma de matriz 
X_train <- train[, c(, , , , )]
X_test <- test[, c(, , , , )]

Y_train <- train[, Pobre]
Y_test <- test[, Pobre]

# Prediccion 
Mod_3_KNN_pred <- knn(X_train, X_test, Y_train, k=3) # k=1

table(Mod_3_KNN_pred, train$Pobre)
library(gmodels)
CrossTable(x=Y_test, y=Mod_3_KNN_pred)
}

### Elastic Net --

### Modelo 4 -
{
  ctrl<- trainControl(method = "cv",
                      number = 5,
                      classProbs = TRUE,
                      savePredictions = T)
  # Modelo
  set.seed(098063)
  Mod_4_EN <- train(Pobre~.,
                  data=train,
                  metric = "Accuracy",
                  method = "glmnet",
                  trControl = ctrl,
                  tuneGrid=expand.grid(
                    alpha = seq(0,1,by=.2),
                    lambda =10^seq(10, -2, length = 10)
                  ))
  # Resultados 
  Mod_4_EN
  # Se escogieron los paramentros ...
}

#### Tree Models ----
{
  library(ISLR2)

  #Asegurarme de que pobre este en formato "si" y "no"
  tree_pobre <- tree(Pobre ~ . , train)
  tree_pobre
  summary(tree_pobre)
  #We see that the training error rate is x%. The Misclassification error rate
  # A small deviance indicates a tree that provides a good fit to the (training) data.
  
  plot(tree_pobre)
  text(tree_pobre, pretty = 0)
  
  ### Evaluating the performance
  set.seed(2)
  train_2 <- sample(1:nrow(train), 200)
  test_2 <- train[-train_2, ]
  tree_pobre_prueba <- tree(Pobre ~ . , train,
                        subset = train_2)
  tree_pred_prueba <- predict(tree_pobre_prueba, test_2,
                       type = "class")
  table(tree_pred_prueba, test_2)
  
  #
  set.seed(7)
  cv.pobre <- cv.tree(tree_pred_prueba, FUN = prune.misclass)
  
  par(mfrow = c(1, 2))
  plot(cv.pobre$Pobre, cv.pobre$dev, type = "b")
  ### Falta
}

#### Bagging ----
{
  library(randomForest)
  
  train_2 <- sample(1:nrow(train), nrow(train) / 2)
  pobre_test <- train[-train_2, "Pobre"]
  
  set.seed(1)
  # configurar el mtry=12 que hace referencia a los predictores 
  # ntree = 25 - podemos complejizar el modelo 
  bag_pobre <- randomForest(Pobre ~ ., data = train,
                             subset = train_2, mtry = 12, importance = TRUE)
  bag_pobre
  
  yhat_bag <- predict(bag_pobre, newdata = train[-train_2, ])
  plot(yhat_bag, pobre_test)
  abline(0, 1)
  
  mean((yhat_bag - pobre_test)^2)
}

#### Random Forest ----
{
  library(randomForest)
  
  train_2 <- sample(1:nrow(train), nrow(train) / 2)
  pobre_test <- train[-train_2, "Pobre"]
  
  
  set.seed(1)
  # configurar el mtry=12 In this case must be the square of the number of variables
  # ntree = 25 - podemos complejizar el modelo 
  rf_pobre <- randomForest(Pobre ~ ., data = train,
                           subset = train_2, mtry = 6, importance = TRUE)
  yhat_rf <- predict(rf_pobre, newdata = train[-train_2, ])
  mean((yhat_rf - pobre_test)^2)
  
  plot(yhat_rf, pobre_test)
  abline(0, 1)
  
  mean((yhat_rf - pobre_test)^2)
  
  # we can view the importance of each variable.
  importance(rf_pobre)
  # The same, but in a diagram
  varImpPlot(rf.boston)
}

# Voy en Boosting




