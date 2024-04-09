##################################################
###################### Data ######################
##################################################

#### 1. Cargar Paquetes ----
{
#se borra la memoria
rm(list = ls())
#se cargan los paquetes
#library(pacman)
# p_load(rio, # importación/exportación de datos
#        tidyverse, # datos ordenados (ggplot y Tidyverse)
#        skimr, # datos de resumen
#        visdat, # visualización de datos faltantes
#        corrplot, # gráficos de correlación
#        stargazer, # tablas/salida a TEX.
#        rvest, # web-scraping
#        readxl,
#        readr, # importar Excel
#        writexl, # exportar Excel
#        boot, # bootstrapping
#        ggpubr, # extensiones de ggplot2
#        WVPlots, # gráficos de variables ponderadas
#        patchwork, # para combinar gráficos
#        gridExtra, # para combinar gráficos
#        ggplot2, # gráficos
#        caret, # para evaluación de modelos predictivos
#        glmnet, # para evaluación de modelos predictivos
#        data.table, # para manipulación de datos
#        class, # El paquete tiene a la funcion de k-neighbours
#        gmodels,
#        MASS,
#        tree,
#        naniar) # missing

require("pacman")
p_load("tidyverse",
       "glmnet",
       "caret",
       "smotefamily",
       "dplyr",
       "dummy",
       "MLeval",
       "pROC") #*MLeval: Machine Learning Model Evaluation


#se define la ruta de trabajo

setwd("C:/Users/Juan/Documents/Problem_set_2_BDML/Data")


load("base_final.RData")
colnames(train_hogares) 
}

#### 2. Selección de variables
{ #Train
  train_hogares <- train_hogares %>% #seleccionar variables
    dplyr::select(-id,
           -Clase, #ya esta cabecera
           -P5010, #¿en cuántos de esos cuartos duermen las personas de este hogar?
           -P5100,#cuando paga por amort (ya esta con ln(cuota)
           -P5140,#arriendo ya esta con ln,
           -Npersug, #no. personas unidad gasto,
           -Ingtotug,
           -Ingtotugarr,
           -Li,
           -Lp,
           -Ingpcug,
           -Ln_Ing_tot_hogar_imp_arr,
           -Ln_Ing_tot_hogar_per_cap,
           -Ln_Ing_tot_hogar,
           -Fex_c)
  
  dummys <- dummy(subset(train_hogares, select = c(Dominio, Depto, Ocup_vivienda, maxEducLevel, 
                                                   Head_EducLevel, Head_Oficio, Head_Ocupacion)))
  dummys <- as.data.frame(apply(dummys,2,function(x){as.numeric(x)}))
  
  train_hogares <- cbind(subset(train_hogares, select = -c(Dominio, Depto, Ocup_vivienda, maxEducLevel, 
                                                           Head_EducLevel, Head_Oficio, Head_Ocupacion)),dummys)
  
}

{ # Test
  test_hogares <- test_hogares %>% #seleccionar variables
    dplyr::select(-Clase, #ya esta cabecera
           -P5010, #¿en cuántos de esos cuartos duermen las personas de este hogar?
           -P5100,#cuando paga por amort (ya esta con ln(cuota)
           -P5140,#arriendo ya esta con ln,
           -Li,
           -Lp,
           -Npersug, #no. personas unidad gasto,
           -Fex_c)
  
  dummys <- dummy(subset(test_hogares, select = c(Dominio, Depto, Ocup_vivienda, maxEducLevel, 
                                                  Head_EducLevel, Head_Oficio, Head_Ocupacion)))
  dummys <- as.data.frame(apply(dummys,2,function(x){as.numeric(x)}))
  
  test_hogares <- cbind(subset(test_hogares, select = -c(Dominio, Depto, Ocup_vivienda, maxEducLevel, 
                                                         Head_EducLevel, Head_Oficio, Head_Ocupacion)),dummys)
  #dejar variables que comparten test y train depsues de crear dummys
  train_hogares <- train_hogares[c(colnames(test_hogares)[2:ncol(test_hogares)],"Pobre")]
  
}

#### 3. Imbalances ----
prop.table(table(train_hogares$Pobre))

# Dividir los datos en conjuntos de entrenamiento (train) y prueba (test)
set.seed(6392) # Para reproducibilidad
train_indices <- as.integer(createDataPartition(train_hogares$Pobre, p = 0.8, list = FALSE))
train <- train_hogares[train_indices, ]
test <- train_hogares[-train_indices, ]
prop.table(table(train$Pobre))
prop.table(table(test$Pobre))

predictors <- colnames(train  %>% dplyr::select(-Pobre))
smote_output <- SMOTE(X = train[predictors],
                      target = train$Pobre)
smote_data <- smote_output$data

table(train$Pobre)
table(smote_data$class)



#### 4. Modelos ----

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
{
  ## Entrenamiento
  
  # Modelo
  library(MASS)
  Mod_2_LDA <- lda(formula = Pobre ~., 
                  data = train)
  Mod_2_LDA
  plot(Mod_2_LDA)

  # Prediccion out-sample
  Mod_2_LDA_pred <- predict(Mod_2_LDA, test)
  names(Mod_2_LDA_pred)
  Mod_2_LDA_pred.class <- Mod_2_LDA_pred$class
  
  # Matriz de confusion
  table(Mod_2_LDA_pred.class, test$Pobre) #aca se puede ajustar el umbral
  mean(Mod_2_LDA_pred.class == test$Pobre)
  
  confusionMatrix(data = Mod_2_LDA_pred.class, 
                  reference = test$Pobre, positive="Yes", mode = "prec_recall")
  #F1 = 0.51
  
}
## Desde acá
#remuestreo hibrido
##smote approach
set.seed(6392)

Mod_2_LDA_smote <- lda(formula = class ~., 
                 data = smote_data)
Mod_2_LDA_smote
plot(Mod_2_LDA_smote)

# Prediccion out-sample
Mod_2_LDA_pred_smote <-  predict(Mod_2_LDA_smote,newdata = test,
                                                      type = "raw")
names(Mod_2_LDA_pred_smote)
head(Mod_2_LDA_pred_smote)
Mod_2_LDA_pred_smote.class <- Mod_2_LDA_pred_smote$class


# Matriz de confusion
table(Mod_2_LDA_pred_smote.class, test$Pobre) #aca se puede ajustar el umbral
mean(Mod_2_LDA_pred_smote.class == test$Pobre)


confusionMatrix(data = Mod_2_LDA_pred_smote.class, 
                reference = test$Pobre, positive="Yes", mode = "prec_recall")
#F1 = 0.65



# Predicción Modelo 2 con Smote
library(stats)
predictSample <- test_hogares   %>% 
  dplyr::mutate(Pobre = predict(Mod_2_LDA_smote, newdata = test_hogares, type = "raw")$class)  %>% 
  dplyr::select(id,Pobre)

predictSample<- predictSample %>% 
  dplyr::mutate(pobre=ifelse(Pobre=="Yes",1,0)) %>% 
  dplyr::select(id,pobre)

write.csv(predictSample,"classification_Lin_Disc_Analysis_smote.csv", row.names = FALSE)

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




