##prueba modelos

rm(list = ls())
require("pacman")
p_load("tidyverse",
       "glmnet",
       "caret",
       "smotefamily",
       "dplyr",
       "dummy",
       "Metrics", # Evaluation Metrics for ML
       "MLeval",#*MLeval: Machine Learning Model Evaluation
       "pROC",
       "ROSE",#remuestreo ROSE
       "ranger") #random forest 

setwd("/Users/camilabeltran/OneDrive/Educación/PEG - UniAndes/BDML/Problem_set_2_BDML/Data")
load("base_final.RData")
colnames(train_hogares) 

#modelo 1 - logit con remuestreo SMOTE F1 = 0.58
#variables: train_hogares <- train_hogares %>% #seleccionar variables
#       select(Dominio, Ocup_vivienda, Nper, maxEducLevel, nocupados, nincapacitados,
#       Cabecera, DormitorXpersona, Head_Mujer, ntrabajo_menores, Pobre)
{
train_hogares <- train_hogares %>% #seleccionar variables
  select(Dominio, Ocup_vivienda, Nper, maxEducLevel, nocupados, nincapacitados,
         Cabecera, DormitorXpersona, Head_Mujer, ntrabajo_menores, Pobre)

dummys <- dummy(train_hogares[,c("Dominio","Ocup_vivienda","maxEducLevel")])
dummys <- as.data.frame(apply(dummys,2,function(x){as.numeric(x)}))

train_hogares <- cbind(train_hogares[c("Nper","nocupados","nincapacitados","Cabecera","DormitorXpersona",
                                       "Head_Mujer","ntrabajo_menores","Pobre")],dummys)

str(train_hogares)

#Clase Ignacio: Imbalance
prop.table(table(train_hogares$Pobre))

# Dividir los datos en conjuntos de entrenamiento (train) y prueba (test)
set.seed(6392) # Para reproducibilidad
train_indices <- as.integer(createDataPartition(train_hogares$Pobre, p = 0.8, list = FALSE))
train <- train_hogares[train_indices, ]
test <- train_hogares[-train_indices, ]
prop.table(table(train$Pobre))
prop.table(table(test$Pobre))

# Sin técnicas de rembalanceo
ctrl<- trainControl(method = "cv",
                    number = 10,
                    classProbs = TRUE,
                    verbose=FALSE,
                    savePredictions = T)
# logit
set.seed(6392)
pobre_logit_orig <- train(Pobre~., 
                           data = train, 
                           method = "glm",
                           trControl = ctrl,
                           family = "binomial")

pobre_logit_orig

test <- test  %>% mutate(pobre_hat_logit_orig=predict(pobre_logit_orig,newdata = test,
                                                      type = "raw"))
confusionMatrix(data = test$pobre_hat_logit_orig, 
                reference = test$Pobre, positive="Yes", mode = "prec_recall")

#F1 = 0.51

#remuestreo hibrido
##smote approach

predictors<-colnames(train  %>% select(-Pobre))
smote_output <- SMOTE(X = train[predictors],
                      target = train$Pobre)
smote_data <- smote_output$data

table(train$Pobre)
table(smote_data$class)

set.seed(6392)

pobre_logit_smote <- train(class~., 
                            data = smote_data, 
                            method = "glm",
                            trControl = ctrl,
                            family = "binomial")

pobre_logit_smote

test<- test  %>% mutate(pobre_hat_logit_smote=predict(pobre_logit_smote,newdata = test,
                                                     type = "raw"))
confusionMatrix(data = test$pobre_hat_logit_smote, 
                reference = test$Pobre, positive="Yes", mode = "prec_recall")

#F1 = 0.58
}

#modelo 2 - elastic net con remuestreo SMOTE F1 = 0.66 Kaggle
{
#modelo 2
train_hogares <- train_hogares %>% #seleccionar variables
         select(-id,
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

test_hogares <- test_hogares %>% #seleccionar variables
  select(-Clase, #ya esta cabecera
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

# Dividir los datos en conjuntos de entrenamiento (train) y prueba (test)
set.seed(6392) # Para reproducibilidad
train_indices <- as.integer(createDataPartition(train_hogares$Pobre, p = 0.8, list = FALSE))
train <- train_hogares[train_indices, ]
test <- train_hogares[-train_indices, ]
prop.table(table(train$Pobre))
prop.table(table(test$Pobre))

predictors <- colnames(train  %>% select(-Pobre))
smote_output <- SMOTE(X = train[predictors],
                      target = train$Pobre)
smote_data <- smote_output$data

table(train$Pobre)
table(smote_data$class)

set.seed(6392)

ctrl<- trainControl(method = "cv",
                    number = 5,
                    classProbs = TRUE,
                    savePredictions = T)

model1 <- train(class~.,
                data=smote_data,
                metric = "Accuracy",
                method = "glmnet",
                trControl = ctrl,
                tuneGrid=expand.grid(
                  alpha = seq(0,1,by=.2),
                  lambda =10^seq(10, -2, length = 10)))

model1

test<- test  %>% mutate(pobre_hat_model1=predict(model1,newdata = test,
                                                      type = "raw"))
confusionMatrix(data = test$pobre_hat_model1, 
                reference = test$Pobre, positive="Yes", mode = "prec_recall")

#F1 = O.66

predictSample <- test_hogares   %>% 
  mutate(Pobre = predict(model1, newdata = test_hogares, type = "raw"))  %>% select(id,Pobre)

predictSample<- predictSample %>% 
  mutate(pobre=ifelse(Pobre=="Yes",1,0)) %>% 
  select(id,pobre)

write.csv(predictSample,"classification_elasticnet_smote.csv", row.names = FALSE)

}

#modelo 3 - model tuning elastic net con remuestreo SMOTE
{
  #modelo 3
  train_hogares <- train_hogares %>% #seleccionar variables
    select(-id,
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
  
  test_hogares <- test_hogares %>% #seleccionar variables
    select(-Clase, #ya esta cabecera
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
  
  # Dividir los datos en conjuntos de entrenamiento (train) y prueba (test)
  set.seed(6392) # Para reproducibilidad
  train_indices <- as.integer(createDataPartition(train_hogares$Pobre, p = 0.8, list = FALSE))
  train <- train_hogares[train_indices, ]
  test <- train_hogares[-train_indices, ]
  prop.table(table(train$Pobre))
  prop.table(table(test$Pobre))
  
  predictors <- colnames(train  %>% select(-Pobre))
  smote_output <- SMOTE(X = train[predictors],
                        target = train$Pobre)
  smote_data <- smote_output$data
  
  table(train$Pobre)
  table(smote_data$class)
  
  multiStats <- function(...) c(twoClassSummary(...), defaultSummary(...), prSummary(...))
  
  ctrl_multiStats<- trainControl(method = "cv",
                                 number = 5,
                                 summaryFunction = multiStats,
                                 classProbs = TRUE,
                                 verbose=FALSE,
                                 savePredictions = T)
  
  lambda <- 10^seq(-1, -4, length = 100)
  
  set.seed(6392)
  
  glm_model_en_f1 <- 
    train(class~.,
    method = "glmnet",
    data = smote_data,
    trControl = ctrl_multiStats,
    tuneGrid = expand.grid(
      alpha = seq(0,1,by=.2),
      lambda =10^seq(10, -2, length = 10)),
    preProcess = c("center", "scale"),
    ## Specify which metric to optimize
    metric = "F"
  )
  
  glm_model_en_f1 
  
  test<- test  %>% mutate(pobre_hat_glm_model_en_f1=predict(glm_model_en_f1,newdata = test,
                                                   type = "raw"))
  confusionMatrix(data = test$pobre_hat_glm_model_en_f1, 
                  reference = test$Pobre, positive="Yes", mode = "prec_recall")
  
  #Alternative cutoffs
  roc_obj_lasso<-roc(response=glm_model_en_f1$pred$obs[glm_model_en_f1$pred$lambda==glm_model_en_f1$bestTune$lambda],
                     predictor=glm_model_en_f1$pred$Yes[glm_model_en_f1$pred$lambda==glm_model_en_f1$bestTune$lambda])
  
  rfThresh_lasso <- coords(roc_obj_lasso, x = "best", best.method = "closest.topleft")
  rfThresh_lasso
  pred_lasso<-factor(ifelse(glm_model_en_f1$pred$Yes[glm_model_en_f1$pred$lambda==glm_model_en_f1$bestTune$lambda]>=rfThresh_lasso$threshold,
                            "Yes","No"),levels=c("Yes","No"))
  confusionMatrix(data = pred_lasso, 
                  reference = glm_model_en_f1$pred$obs[glm_model_en_f1$pred$lambda==glm_model_en_f1$bestTune$lambda], 
                  positive="Yes", mode = "prec_recall")
  
  prec_recall<-data.frame(coords(roc_obj_lasso, seq(0,1,length=100), ret=c("threshold", "precision", "recall")))
  prec_recall<- prec_recall  %>% mutate(F1=(2*precision*recall)/(precision+recall))
  prec_recall$threshold[which.max(prec_recall$F1)]
  pred_lasso_F1<-factor(ifelse(glm_model_en_f1$pred$Yes[glm_model_en_f1$pred$lambda==glm_model_en_f1$bestTune$lambda]>=prec_recall$threshold[which.max(prec_recall$F1)],
                               "Yes","No"),levels=c("Yes","No"))
  confusionMatrix(data = pred_lasso_F1, 
                  reference = glm_model_en_f1$pred$obs[glm_model_en_f1$pred$lambda==glm_model_en_f1$bestTune$lambda], 
                  positive="Yes", mode = "prec_recall")
}

#modelo 4 - elastic net con remuestreo ROSE 
{
  train_hogares <- train_hogares %>% #seleccionar variables
    select(-id,
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
  
  test_hogares <- test_hogares %>% #seleccionar variables
    select(-Clase, #ya esta cabecera
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
  
  # Dividir los datos en conjuntos de entrenamiento (train) y prueba (test)
  set.seed(6392) # Para reproducibilidad
  train_indices <- as.integer(createDataPartition(train_hogares$Pobre, p = 0.8, list = FALSE))
  train <- train_hogares[train_indices, ]
  test <- train_hogares[-train_indices, ]
  prop.table(table(train$Pobre))
  prop.table(table(test$Pobre))
  
  #remuestreo con ROSE
  set.seed(6392)
  rose_train <- ROSE(Pobre~., data  = train)$data                         
  table(rose_train$Pobre) 
  
  #logit
  ctrl<- trainControl(method = "cv",
                      number = 5,
                      classProbs = TRUE,
                      savePredictions = T)
  
  pobre_logit_rose <- train(Pobre~., 
                          data = rose_train, 
                          method = "glm",
                          trControl = ctrl,
                          family = "binomial")
  
  pobre_logit_rose
  
  test <- test  %>% mutate(pobre_hat_logit_rose=predict(pobre_logit_rose,newdata = test,
                                                   type = "raw"))
  confusionMatrix(data = test$pobre_hat_logit_rose, 
                  reference = test$Pobre, positive="Yes", mode = "prec_recall")
  
  #F1 = 0.61

  #elasticnet
  pobre_en_rose <- train(Pobre~.,
                  data=rose_train,
                  metric = "Accuracy",
                  method = "glmnet",
                  trControl = ctrl,
                  tuneGrid=expand.grid(
                    alpha = seq(0,1,by=.2),
                    lambda =10^seq(10, -2, length = 10)))
  
  pobre_en_rose
  
  test<- test  %>% mutate(pobre_hat_en_rose=predict(pobre_en_rose,newdata = test,
                                                   type = "raw"))
  confusionMatrix(data = test$pobre_hat_en_rose, 
                  reference = test$Pobre, positive="Yes", mode = "prec_recall")
  
  #F1 = O.61
  
  # Predicción Kaggle 1
  predictSample <- test_hogares   %>% 
    mutate(Pobre = predict(model1, newdata = test_hogares, type = "raw"))  %>% select(id,Pobre)
  
  predictSample<- predictSample %>% 
    mutate(pobre=ifelse(Pobre=="Yes",1,0)) %>% 
    select(id,pobre)
  
  write.csv(predictSample,"classification_elasticnet_smote.csv", row.names = FALSE)
  
}

#modelo 5 - elastic net con remuestreo SMOTE (var de conyuge y % de personas)
{
  #modelo 5

  train_hogares <- train_hogares %>%
    mutate(
      prop_mujeres <- nmujeres/Nper,
      prop_ocupados <- nocupados/Nper,
      prop_incapacitados <- nincapacitados/Nper,
      soltero_menores <- ifelse(Head_Mujer==0&nmenores_5>0&nmenores_6_11>0&nmenores_12_17>0&nconyuge==0,0,1),
      soltera_menores <- ifelse(Head_Mujer==1&nmenores_5>0&nmenores_6_11>0&nmenores_12_17>0&nconyuge==0,0,1))
  
  test_hogares <- test_hogares %>%
    mutate(
      prop_mujeres <- nmujeres/Nper,
      prop_ocupados <- nocupados/Nper,
      prop_incapacitados <- nincapacitados/Nper,
      soltero_menores <- ifelse(Head_Mujer==0&nmenores_5>0&nmenores_6_11>0&nmenores_12_17>0&nconyuge==0,0,1),
      soltera_menores <- ifelse(Head_Mujer==1&nmenores_5>0&nmenores_6_11>0&nmenores_12_17>0&nconyuge==0,0,1))
  
  train_hogares <- train_hogares %>% #seleccionar variables
    select(-id,
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
           -Fex_c,
           -nmujeres,
           -nocupados,
           -nincapacitados,
           -nconyuge)
  
  dummys <- dummy(subset(train_hogares, select = c(Dominio, Depto, Ocup_vivienda, maxEducLevel, 
                                                   Head_EducLevel, Head_Oficio, Head_Ocupacion,
                                                   Cony_EducLevel, Cony_Oficio, Cony_Ocupacion)))
  
  dummys <- as.data.frame(apply(dummys,2,function(x){as.numeric(x)}))
  
  train_hogares <- cbind(subset(train_hogares, select = -c(Dominio, Depto, Ocup_vivienda, maxEducLevel, 
                                                           Head_EducLevel, Head_Oficio, Head_Ocupacion,
                                                           Cony_EducLevel, Cony_Oficio, Cony_Ocupacion)),dummys)
  
  test_hogares <- test_hogares %>% #seleccionar variables
    select(-Clase, #ya esta cabecera
           -P5010, #¿en cuántos de esos cuartos duermen las personas de este hogar?
           -P5100,#cuando paga por amort (ya esta con ln(cuota)
           -P5140,#arriendo ya esta con ln,
           -Li,
           -Lp,
           -Npersug, #no. personas unidad gasto,
           -Fex_c,
           -nmujeres,
           -nocupados,
           -nincapacitados,
           -nconyuge)
  
  dummys <- dummy(subset(test_hogares, select = c(Dominio, Depto, Ocup_vivienda, maxEducLevel, 
                                                  Head_EducLevel, Head_Oficio, Head_Ocupacion,
                                                  Cony_EducLevel, Cony_Oficio, Cony_Ocupacion)))
  dummys <- as.data.frame(apply(dummys,2,function(x){as.numeric(x)}))
  
  test_hogares <- cbind(subset(test_hogares, select = -c(Dominio, Depto, Ocup_vivienda, maxEducLevel, 
                                                         Head_EducLevel, Head_Oficio, Head_Ocupacion,
                                                         Cony_EducLevel, Cony_Oficio, Cony_Ocupacion)),dummys)
  
  #dejar variables que comparten test y train depsues de crear dummys
  train_hogares <- train_hogares[c(colnames(test_hogares)[2:ncol(test_hogares)],"Pobre")]
  
  # Dividir los datos en conjuntos de entrenamiento (train) y prueba (test)
  set.seed(6392) # Para reproducibilidad
  train_indices <- as.integer(createDataPartition(train_hogares$Pobre, p = 0.8, list = FALSE))
  train <- train_hogares[train_indices, ]
  test <- train_hogares[-train_indices, ]
  prop.table(table(train$Pobre))
  prop.table(table(test$Pobre))
  
  predictors <- colnames(train  %>% select(-Pobre))
  smote_output <- SMOTE(X = train[predictors],
                        target = train$Pobre)
  smote_data <- smote_output$data
  
  table(train$Pobre)
  table(smote_data$class)
  
  set.seed(6392)
  
  ctrl<- trainControl(method = "cv",
                      number = 5,
                      classProbs = TRUE,
                      savePredictions = T)
  
  model1 <- train(class~.,
                  data=smote_data,
                  metric = "Accuracy",
                  method = "glmnet",
                  trControl = ctrl,
                  tuneGrid=expand.grid(
                    alpha = seq(0,1,by=.2),
                    lambda =10^seq(10, -2, length = 10)))
  
  model1
  
  test<- test  %>% mutate(pobre_hat_model1=predict(model1,newdata = test,
                                                   type = "raw"))
  confusionMatrix(data = test$pobre_hat_model1, 
                  reference = test$Pobre, positive="Yes", mode = "prec_recall")
  
  #F1 = O.66
  
  predictSample <- test_hogares   %>% 
    mutate(Pobre = predict(model1, newdata = test_hogares, type = "raw"))  %>% select(id,Pobre)
  
  predictSample<- predictSample %>% 
    mutate(pobre=ifelse(Pobre=="Yes",1,0)) %>% 
    select(id,pobre)
  
  write.csv(predictSample,"classification_elasticnet_smote_mas_variables.csv", row.names = FALSE)
  
}

#buscar variables mas importantes
{
train_hogares <- train_hogares %>% #seleccionar variables
  select(-id,
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

test_hogares <- test_hogares %>% #seleccionar variables
  select(-Clase, #ya esta cabecera
         -P5010, #¿en cuántos de esos cuartos duermen las personas de este hogar?
         -P5100,#cuando paga por amort (ya esta con ln(cuota)
         -P5140,#arriendo ya esta con ln,
         -Li,
         -Lp,
         -Npersug, #no. personas unidad gasto,
         -Fex_c)


model_logit <- glm(Pobre ~ ., data = train_hogares, family = "binomial")
summary(model_logit)
}

#random forest con cross-validation 1 (algunas variables)
{
train_hogares <- train_hogares %>% 
    select(Pobre,Dominio,Depto,N_cuartos_hog,Nper,nmenores_5
           ,nmenores_6_11,nmenores_12_17,nocupados,nincapacitados,ntrabajo_menores,
           Head_Mujer,Head_Afiliado_SS,Head_exper_ult_trab,Head_Rec_alimento,
           Head_Rec_subsidio,Head_Rec_vivienda,Head_Ocupacion,Head_Segundo_trabajo,
           DormitorXpersona,Ln_Cuota,Ln_Pago_arrien) %>% 
    mutate(
      Head_Mujer <- factor(Head_Mujer),
      Head_Afiliado_SS <- factor(Head_Afiliado_SS),
      Head_exper_ult_trab <- factor(Head_exper_ult_trab),
      Head_Rec_alimento <- factor(Head_Rec_alimento),
      Head_Rec_subsidio <- factor(Head_Rec_subsidio),
      Head_Rec_vivienda <- factor(Head_Rec_vivienda),
      Head_Ocupacion <- factor(Head_Ocupacion),
      Head_Segundo_trabajo <- factor(Head_Segundo_trabajo))
  
  test_hogares <- test_hogares %>% 
    select(id,Dominio,Depto,N_cuartos_hog,Nper,nmenores_5
           ,nmenores_6_11,nmenores_12_17,nocupados,nincapacitados,ntrabajo_menores,
           Head_Mujer,Head_Afiliado_SS,Head_exper_ult_trab,Head_Rec_alimento,
           Head_Rec_subsidio,Head_Rec_vivienda,Head_Ocupacion,Head_Segundo_trabajo,
           DormitorXpersona,Ln_Cuota,Ln_Pago_arrien) %>% 
    mutate(
      Head_Mujer <- factor(Head_Mujer),
      Head_Afiliado_SS <- factor(Head_Afiliado_SS),
      Head_exper_ult_trab <- factor(Head_exper_ult_trab),
      Head_Rec_alimento <- factor(Head_Rec_alimento),
      Head_Rec_subsidio <- factor(Head_Rec_subsidio),
      Head_Rec_vivienda <- factor(Head_Rec_vivienda),
      Head_Ocupacion <- factor(Head_Ocupacion),
      Head_Segundo_trabajo <- factor(Head_Segundo_trabajo))  

  fiveStats <- function(...) c(twoClassSummary(...), defaultSummary(...))
  ctrl<- trainControl(method = "cv",
                      number = 5,
                      summaryFunction = fiveStats,
                      classProbs = TRUE,
                      verbose=FALSE,
                      savePredictions = T)
  
  mtry_grid<-expand.grid(mtry =c(5,10,20), # 8 inclueye bagging
                         min.node.size= c(100,200,300), #controla la complejidad del arbol
                         splitrule= 'gini') #splitrule fija en gini. 
  mtry_grid
  
  cv_RForest <- train(Pobre~Dominio+Depto+N_cuartos_hog+Nper+nmenores_5
                      +nmenores_6_11+nmenores_12_17+nocupados+nincapacitados+ntrabajo_menores+
                      Head_Mujer+Head_Afiliado_SS+Head_exper_ult_trab+Head_Rec_alimento+
                      Head_Rec_subsidio+Head_Rec_vivienda+Head_Ocupacion+Head_Segundo_trabajo+
                      DormitorXpersona+Ln_Cuota+Ln_Pago_arrien, 
                      data = train_hogares, 
                      method = "ranger",
                      trControl = ctrl,
                      metric="ROC",
                      tuneGrid = mtry_grid,
                      ntree=500)
  
  cv_RForest
  cv_RForest$finalModel
  
  predictSample <- test_hogares   %>% 
    mutate(Pobre = predict(cv_RForest, newdata = test_hogares, type = "raw"))  %>% select(id,Pobre)
  
  predictSample<- predictSample %>% 
    mutate(pobre=ifelse(Pobre=="Yes",1,0)) %>% 
    select(id,pobre)
  
  write.csv(predictSample,"classification_random_forest1.csv", row.names = FALSE)

  #aucval_rf <- Metrics::auc(actual = Pobre,predicted =rf_pred[,2])
  #aucval_rf
  
  
}

#random forest con cross-validation 2 (todas variables)
{
  load("base_final.RData")
  train_hogares <- train_hogares %>% #seleccionar variables
    select(-id,
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
  
  test_hogares <- test_hogares %>% #seleccionar variables
    select(-Clase, #ya esta cabecera
           -P5010, #¿en cuántos de esos cuartos duermen las personas de este hogar?
           -P5100,#cuando paga por amort (ya esta con ln(cuota)
           -P5140,#arriendo ya esta con ln,
           -Li,
           -Lp,
           -Npersug, #no. personas unidad gasto,
           -Fex_c)
  
  train_hogares <- train_hogares %>% 
    mutate(
      Head_Mujer <- factor(Head_Mujer),
      Depto <- factor(Depto),
      Head_ocupado <- factor(Head_ocupado),
      Head_Reg_subs_salud <- factor(Head_Reg_subs_salud),
      Head_Afiliado_SS <- factor(Head_Afiliado_SS),
      Head_Rec_alimento <- factor(Head_Rec_alimento),
      Head_Rec_subsidio <- factor(Head_Rec_subsidio),
      Head_Cot_pension <- factor(Head_Cot_pension),
      Head_Rec_vivienda <- factor(Head_Rec_vivienda),
      Head_Ocupacion <- factor(Head_Ocupacion),
      Head_Segundo_trabajo <- factor(Head_Segundo_trabajo),
      Head_Nivel_formalidad <- factor(Head_Nivel_formalidad),
      Head_Oficio <- factor(Head_Oficio),
      Head_Primas <- factor(Head_Primas),
      Head_Bonificaciones <- factor(Head_Bonificaciones),
      Head_Segundo_trabajo <- factor(Head_Segundo_trabajo),
      Cabecera <- factor(Cabecera))  
  
  test_hogares <- test_hogares %>% 
    mutate(
      Head_Mujer <- factor(Head_Mujer),
      Depto <- factor(Depto),
      Head_ocupado <- factor(Head_ocupado),
      Head_Reg_subs_salud <- factor(Head_Reg_subs_salud),
      Head_Afiliado_SS <- factor(Head_Afiliado_SS),
      Head_Rec_alimento <- factor(Head_Rec_alimento),
      Head_Rec_subsidio <- factor(Head_Rec_subsidio),
      Head_Cot_pension <- factor(Head_Cot_pension),
      Head_Rec_vivienda <- factor(Head_Rec_vivienda),
      Head_Ocupacion <- factor(Head_Ocupacion),
      Head_Segundo_trabajo <- factor(Head_Segundo_trabajo),
      Head_Nivel_formalidad <- factor(Head_Nivel_formalidad),
      Head_Oficio <- factor(Head_Oficio),
      Head_Primas <- factor(Head_Primas),
      Head_Bonificaciones <- factor(Head_Bonificaciones),
      Head_Segundo_trabajo <- factor(Head_Segundo_trabajo),
      Cabecera <- factor(Cabecera))  
  
  fiveStats <- function(...) c(twoClassSummary(...), defaultSummary(...))
  ctrl<- trainControl(method = "cv",
                      number = 5,
                      summaryFunction = fiveStats,
                      classProbs = TRUE,
                      verbose=FALSE,
                      savePredictions = T)
  
  mtry_grid<-expand.grid(mtry =c(10,15,30), # 8 inclueye bagging
                         min.node.size= c(25,50,100), #controla la complejidad del arbol
                         splitrule= 'gini') #splitrule fija en gini. 
  mtry_grid
  
  cv_RForest <- train(Pobre~., 
                      data = train_hogares, 
                      method = "ranger",
                      trControl = ctrl,
                      metric="ROC",
                      tuneGrid = mtry_grid,
                      ntree=500)
  
  cv_RForest
  cv_RForest$finalModel
  
  predictSample <- test_hogares   %>% 
    mutate(Pobre = predict(cv_RForest, newdata = test_hogares, type = "raw"))  %>% select(id,Pobre)
  
  predictSample<- predictSample %>% 
    mutate(pobre=ifelse(Pobre=="Yes",1,0)) %>% 
    select(id,pobre)
  
  write.csv(predictSample,"classification_random_forest2.csv", row.names = FALSE)
  
  #aucval_rf <- Metrics::auc(actual = Pobre,predicted =rf_pred[,2])
  #aucval_rf
}

#random forest con cv 3, todas variables y SMOTE 
#Kaggle puntaje = 0.63
{
  load("base_final.RData")
  train_hogares <- train_hogares %>% #seleccionar variables
    select(-id,
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
  
  test_hogares <- test_hogares %>% #seleccionar variables
    select(-Clase, #ya esta cabecera
           -P5010, #¿en cuántos de esos cuartos duermen las personas de este hogar?
           -P5100,#cuando paga por amort (ya esta con ln(cuota)
           -P5140,#arriendo ya esta con ln,
           -Li,
           -Lp,
           -Npersug, #no. personas unidad gasto,
           -Fex_c)

  dummys <- dummy(subset(train_hogares, select = c(Dominio, Depto, Ocup_vivienda, maxEducLevel, 
                                                   Head_EducLevel, Head_Oficio, Head_Ocupacion)))
  dummys <- as.data.frame(apply(dummys,2,function(x){as.numeric(x)}))
  
  train_hogares <- cbind(subset(train_hogares, select = -c(Dominio, Depto, Ocup_vivienda, maxEducLevel, 
                                                           Head_EducLevel, Head_Oficio, Head_Ocupacion)),dummys)
  
  dummys <- dummy(subset(test_hogares, select = c(Dominio, Depto, Ocup_vivienda, maxEducLevel, 
                                                  Head_EducLevel, Head_Oficio, Head_Ocupacion)))
  dummys <- as.data.frame(apply(dummys,2,function(x){as.numeric(x)}))
  
  test_hogares <- cbind(subset(test_hogares, select = -c(Dominio, Depto, Ocup_vivienda, maxEducLevel, 
                                                         Head_EducLevel, Head_Oficio, Head_Ocupacion)),dummys)
  
  #dejar variables que comparten test y train despues de crear dummys
  train_hogares <- train_hogares[c(colnames(test_hogares)[2:ncol(test_hogares)],"Pobre")]
  
  #dejar variables que comparten test y train despues de crear dummys
  train_hogares <- train_hogares[c(colnames(test_hogares)[2:ncol(test_hogares)],"Pobre")]
  
  # Dividir los datos en conjuntos de entrenamiento (train) y prueba (test)
  set.seed(6392) # Para reproducibilidad
  train_indices <- as.integer(createDataPartition(train_hogares$Pobre, p = 0.8, list = FALSE))
  train <- train_hogares[train_indices, ]
  test <- train_hogares[-train_indices, ]
  prop.table(table(train$Pobre))
  prop.table(table(test$Pobre))
  
  predictors <- colnames(train  %>% select(-Pobre))
  smote_output <- SMOTE(X = train[predictors],
                        target = train$Pobre)
  smote_data <- smote_output$data
  
  table(train$Pobre)
  table(smote_data$class)
  
  fiveStats <- function(...) c(twoClassSummary(...), defaultSummary(...))
  ctrl<- trainControl(method = "cv",
                      number = 5,
                      summaryFunction = fiveStats,
                      classProbs = TRUE,
                      verbose=FALSE,
                      savePredictions = T)
  
  mtry_grid<-expand.grid(mtry =c(10,30,50,100), # 8 inclueye bagging
                         min.node.size= c(50,100,200), #controla la complejidad del arbol
                         splitrule= 'gini') #splitrule fija en gini. 
  
  model1 <- train(class~., 
                      data = smote_data, 
                      method = "ranger",
                      trControl = ctrl,
                      metric="ROC",
                      tuneGrid = mtry_grid,
                      ntree=500)
  
  model1
  model1$finalModel
  
  test<- test  %>% mutate(pobre_hat_model1=predict(model1,newdata = test,
                                                   type = "raw"))
  confusionMatrix(data = test$pobre_hat_model1, 
                  reference = test$Pobre, positive="Yes", mode = "prec_recall")
  
  #F1 = O.65
  
  predictSample <- test_hogares   %>% 
    mutate(Pobre = predict(model1, newdata = test_hogares, type = "raw"))  %>% select(id,Pobre)
  
  predictSample<- predictSample %>% 
    mutate(pobre=ifelse(Pobre=="Yes",1,0)) %>% 
    select(id,pobre)
  
  #Kaggle puntaje = 0.63
  write.csv(predictSample,"classification_random_forest3.csv", row.names = FALSE)
}