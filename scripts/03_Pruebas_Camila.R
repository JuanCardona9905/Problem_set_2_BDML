##pruebas modelos

#modelo 1 - logit con remuestreo SMOTE F1 = 0.58
#variables: train_hogares <- train_hogares %>% #seleccionar variables
#       select(Dominio, Ocup_vivienda, Nper, maxEducLevel, nocupados, nincapacitados,
#       Cabecera, DormitorXpersona, Head_Mujer, ntrabajo_menores, Pobre)
{
load("base_final.RData")  
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

# Define una función de resumen personalizada que incluya F1
fiveStats <- function(data, lev = NULL, model = NULL) {
  out <- twoClassSummary(data, lev = lev, model = model)
  out$F1 <- F1_Score(data$obs, data$pred)
  out
}

# Función para calcular F1 Score
F1_Score <- function(actual, predicted) {
  cm <- confusionMatrix(actual, predicted)
  f1 <- ifelse(sum(cm$table) > 0, cm$byClass["F1"], 0)
  f1
}

# Configuración de control para el entrenamiento del modelo
ctrl <- trainControl(
  method = "cv",
  number = 5,
  summaryFunction = fiveStats,
  classProbs = TRUE, 
  verbose = FALSE,
  savePredictions = TRUE
)

# Entrenamiento del modelo con métrica "F1"
model1 <- train(
  class ~ .,
  data = smote_data,
  metric = "F1",  # Utiliza F1 para la evaluación
  method = "glmnet",
  trControl = ctrl,
  tuneGrid = expand.grid(
    alpha = seq(0, 1, by = 0.2),
    lambda = 10^seq(10, -2, length = 10)
  )
)

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
}

#modelo 4 - elastic net con remuestreo ROSE 
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
  
}

#buscar variables mas importantes
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


model_logit <- glm(Pobre ~ ., data = train_hogares, family = "binomial")
summary(model_logit)
}

#random forest con cross-validation 1 (algunas variables)
{
  load("base_final.RData")
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
}

#random forest con cv 3, todas variables y SMOTE 
#Kaggle puntaje = 0.63
#Type:                             Probability estimation 
#  Number of trees:                  500 
#  Sample size:                      184809 
#  Number of independent variables:  187 
#  Mtry:                             30 
#  Target node size:                 50 
#  Variable importance mode:         none 
#  Splitrule:                        gini 
#  OOB prediction error (Brier s.):  0.06681718   
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

#xgboosting con , todas variables y SMOTE 
#Kaggle puntaje = 0.67
#Tuning parameter 'nrounds' was held constant at a value of 500
#Tuning parameter 'max_depth' was held constant at
#a value of 4
#Tuning parameter 'gamma' was held constant at a value of 0
#Tuning parameter 'min_child_weight'
#was held constant at a value of 50
#Tuning parameter 'subsample' was held constant at a value of 0.4
#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were nrounds = 500, max_depth = 4, eta = 0.25, gamma = 0, colsample_bytree
#= 0.33, min_child_weight = 50 and subsample = 0.4.
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
  
  fitControl<-trainControl(method ="cv",
                           number=5)
  
  grid_xbgoost <- expand.grid(nrounds = c(500),
                              max_depth = c(4), 
                              eta = c(0.01,0.25,0.5), 
                              gamma = c(0), 
                              min_child_weight = c(50),
                              colsample_bytree = c(0.33,0.66),
                              subsample = c(0.4))
  set.seed(6392)
  model1 <- train(class~.,
                  data=smote_data,
                  method = "xgbTree", 
                  trControl = fitControl,
                  tuneGrid=grid_xbgoost)        
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
  
  #Kaggle puntaje = 0.67
  #write.csv(predictSample,"classification_xgboosting.csv", row.names = FALSE)
}

#xgboosting con todas las variables (indirecta)
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
           #-Ingtotugarr,
           -Li,
           -Lp,
           -Ingpcug,
           -Ln_Ing_tot_hogar_imp_arr,
           -Ln_Ing_tot_hogar_per_cap,
           -Ln_Ing_tot_hogar,
           -Fex_c,
           -Pobre)
  
  test_hogares <- test_hogares %>% #seleccionar variables
    select(-Clase, #ya esta cabecera
           -P5010, #¿en cuántos de esos cuartos duermen las personas de este hogar?
           -P5100,#cuando paga por amort (ya esta con ln(cuota)
           -P5140,#arriendo ya esta con ln,
           -Li,
           #-Lp,
           #-Npersug, #no. personas unidad gasto,
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
  
  fitControl<-trainControl(method ="cv",
                           number=5)
  
  grid_xbgoost <- expand.grid(nrounds = c(500),
                              max_depth = c(4), 
                              eta = c(0.01,0.25,0.5), 
                              gamma = c(0), 
                              min_child_weight = c(50),
                              colsample_bytree = c(0.33,0.66),
                              subsample = c(0.4))
  set.seed(6392)
  model1 <- train(Ingtotugarr~.,
                  data=train_hogares,
                  method = "xgbTree", 
                  trControl = fitControl,
                  tuneGrid=grid_xbgoost)        
  model1
  
  predictSample <- test_hogares   %>% 
    mutate(ingreso_predict= predict(model1, newdata = test_hogares))  %>% select(id,ingreso_predict,Lp,Nper)
  
  predictSample$pobre <- ifelse(predictSample$ingreso_predict <= predictSample$Lp*predictSample$Nper, 1 ,0)
  predictSample<- predictSample %>% 
    select(id,pobre)

  write.csv(predictSample,"Regresion_indirecta_xgboosting_cami.csv", row.names=FALSE)
  
}
