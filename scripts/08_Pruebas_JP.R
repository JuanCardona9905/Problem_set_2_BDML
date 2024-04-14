##################################################
######### Modelos de clasificación  ##############
##################################################

#### 1. Cargar Paquetes ----
{
#se borra la memoria
rm(list = ls())

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
{
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
  
}

#### 4. Modelos ----
### Logistic Regression --

### Linear Discriminant Analysis --
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
{
# Es necesario los datos en forma de matriz 

# Prediccion 
library(caret)
library(class)
Mod_3_KNN_pred <- knn(train[,c(predictors)], test[,c(predictors)], train$Pobre, k=15) # k=1

knnModel <- train(
  Pobre ~ ., 
  data = test, 
  method = "knn", 
  trControl = trainControl(method = "cv"), 
  tuneGrid = data.frame(k = c(7,15))
)

best_model<- knn3(
  not_fully_paid ~ .,
  data = test,
  k = knnModel$bestTune$k
)

predictions <- predict(best_model, testTransformed,type = "class")
# Calculate confusion matrix
cm <- confusionMatrix(predictions, testTransformed$not_fully_paid)
cm

predictSample <- test_hogares   %>% 
  mutate(Pobre = predict(knnModel, newdata = test_hogares, type = "raw"))  %>% select(id,Pobre)

predictSample<- predictSample %>% 
  mutate(pobre=ifelse(Pobre=="Yes",1,0)) %>% 
  select(id,pobre)

write.csv(predictSample,"KNN_Model.csv", row.names = FALSE)







view(Mod_3_KNN_pred)
view(train$Pobre)

table(Mod_3_KNN_pred, test$Pobre)
library(gmodels)
CrossTable(x=test$Pobre, y=Mod_3_KNN_pred)

confusionMatrix(data = Mod_3_KNN_pred, 
                reference = test$Pobre, positive="Yes", mode = "prec_recall")




}

