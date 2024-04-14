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

#################################################
######## Modelo indirectos lineales #############
###################### ##########################

######### A. Modelo simple ######################

#### 1. Cargar los paquetes ----
{
  install.packages("pacman")
  
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
  
  
  setwd("C:/Users/Juan/Documents/Problem_set_2_BDML/Data")
  
  
  load("base_final.RData")
  colnames(train_hogares) 
}

#### 2. Selección de variables ----
{ #Train
  train_hogares <- train_hogares %>% #seleccionar variables
    dplyr::mutate(Ln_Ing_hog = log(Ingtotugarr),
                  PerXCuarto = Nper/N_cuartos_hog) %>% 
    dplyr::select(-id,
                  -Clase, #ya esta cabecera
                  -P5010, #¿en cuántos de esos cuartos duermen las personas de este hogar?
                  -P5100,#cuando paga por amort (ya esta con ln(cuota)
                  -P5140,#arriendo ya esta con ln,
                  -Ingtotug,
                  -Ingtotugarr,
                  -Li,
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
  # Filtro de logaritmo
  train_hogares <- train_hogares %>%
    filter(Ln_Ing_hog != "-Inf")
  
  summary(train_hogares$Ln_Ing_hog)
  }

{ # Test
  test_hogares <- test_hogares %>% #seleccionar variables
    dplyr::mutate(PerXCuarto = Nper/N_cuartos_hog) %>% 
    dplyr::select(-Clase, #ya esta cabecera
                  -P5010, #¿en cuántos de esos cuartos duermen las personas de este hogar?
                  -P5100,#cuando paga por amort (ya esta con ln(cuota)
                  -P5140,#arriendo ya esta con ln,
                  -Li,
                  -Fex_c)
  
  dummys <- dummy(subset(test_hogares, select = c(Dominio, Depto, Ocup_vivienda, maxEducLevel, 
                                                  Head_EducLevel, Head_Oficio, Head_Ocupacion)))
  dummys <- as.data.frame(apply(dummys,2,function(x){as.numeric(x)}))
  
  test_hogares <- cbind(subset(test_hogares, select = -c(Dominio, Depto, Ocup_vivienda, maxEducLevel, 
                                                         Head_EducLevel, Head_Oficio, Head_Ocupacion)),dummys)
  #dejar variables que comparten test y train depsues de crear dummys
  train_hogares <- train_hogares[c(colnames(test_hogares)[2:ncol(test_hogares)],"Ln_Ing_hog", "Pobre")]
}

#### 3. Imbalances ----
{
   view(train_hogares$Ln_Ing_hog)
  
  # Dividir los datos en conjuntos de entrenamiento (train) y prueba (test)
  set.seed(6392) # Para reproducibilidad
  train_indices <- as.integer(createDataPartition(train_hogares$Ln_Ing_hog, p = 0.8, list = FALSE))
  train <- train_hogares[train_indices, ]
  test <- train_hogares[-train_indices, ]
  
  summary(train$Ln_Ing_hog) 
  summary(test$Ln_Ing_hog) 
  
  predictors <- colnames(train  %>% dplyr::select(-Ln_Ing_hog))
  colnames(train)
  table(train$Cabecera)
  }


### 4. Verificacion de Subset Selection ----
{ # Seleccion Variables - modelo corto
  train <- train %>% 
    dplyr::select(Ln_Ing_hog, PerXCuarto, nocupados, Head_Mujer, Head_Cot_pension, Head_Rec_subsidio, Npersug, Lp, Pobre)
  
  test <- test %>% 
    dplyr::select(Ln_Ing_hog, PerXCuarto, nocupados, Head_Mujer, Head_Cot_pension, Head_Rec_subsidio, Npersug, Lp, Pobre)
}

{ # Modelo 
  model_form <- Ln_Ing_hog ~ PerXCuarto + 
    nocupados +
    as.factor(Head_Mujer)+
    Head_Cot_pension+
    Head_Rec_subsidio
}

{ # Best Subset Selection Model
  p_load(rio, # import/export data
         tidyverse, # tidy-data
         caret, # For predictive model assessment
         leaps)#,    #for subset  model selection
  #skimr)    # summary data
  
  # Best subset selection model
  bestsub_model <- regsubsets(model_form, ## formula
                              data = train, ## data frame Note we are using the training sample.
                              nvmax = 11 ## run the first 11 models
  )  ## apply Forward Stepwise Selection
  
  summary(bestsub_model)
}

{ # Armar el modelo
  
  max_nvars= bestsub_model[["np"]]-1  ## minus one because it counts the intercept.
  max_nvars
  
  set.seed(3963)
  
  ## create the predict Function for the object regsubsets
  # 1  
  predict.regsubsets<- function (object , newdata , id, ...) {
    form<- model_form
    mat <- model.matrix (form , newdata) ## model matrix in the test data
    coefi <- coef(object , id = id) ## coefficient for the best model with id vars
    xvars <- names (coefi)  ## variables in the model
    mat[, xvars] %*% coefi  ## prediction 
  }
  
  # 2 
  k <- 5
  n <- nrow (train)
  folds <- sample (rep (1:k, length = n))
  cv.errors <- matrix (NA, k, max_nvars,
                       dimnames = list (NULL , paste (1:max_nvars)))
  
  # 3 
  for (j in 1:k) {
    best_fit <- regsubsets(model_form,
                           data = train[folds != j, ],
                           nvmax = max_nvars)
    for (i in 1:max_nvars) {
      pred <- predict(best_fit , train[folds == j, ], id = i)
      cv.errors[j, i] <-
        mean((train$Ln_Ing_hog[folds == j] - pred)^2)
    }}
}

{ # Medidas de ajuste
  mean.cv.errors <- apply (cv.errors , 2, mean)
  mean.cv.errors
  
  which.min (mean.cv.errors) # El mejor modelo es de las cinco variables
  
  mean.cv.errors.df <- as.data.frame(mean.cv.errors)
  mean.cv.errors.df <- mean.cv.errors.df %>% 
    mutate(index_df = seq(from = 1, to = 5, by =1))
  
  # Grafica de error por variable 
  plot (mean.cv.errors , type = "b")
  ggplot(data=mean.cv.errors.df, aes(x=index_df, y=mean.cv.errors, group=1)) +
    geom_line()+
    geom_point()
}

### 5. Elastic Net ---- 
{
  # 1.
  tuneGrid<- expand.grid(alpha= seq(0,1, 0.05), # between 0 and 1. 
                         lambda=seq(0.5, 1.5, 0.5) ) 
  
  fitControl <- trainControl( 
    method = "cv",
    number = 10) ##  10 fold CV
  
  
  # 2. 
  ENet<-train(model_form,
              data=train,
              method = 'glmnet', 
              trControl = fitControl,
              tuneGrid = tuneGrid)  #specify the grid 
  ENet # el mejor es 
  which.min(ENet$results[,3]) # El primer modelo alpha 0 y lambda 0.5
  plot(ENet)
  
  # 3. 
  tuneGrid<- expand.grid(alpha= seq(0.05,0.95, 0.05), # between 0 and 1. 
                         lambda=seq(0.1, 2, 0.1) ) 
  ENet<-train(model_form,
              data=train,
              method = 'glmnet', 
              trControl = fitControl,
              tuneGrid = tuneGrid)
  plot(ENet)
  
  #4. 
  ENet$bestTune #alpha 0.05 - #lambda 0.1
  
  #5.
  # El mejor modelo
  Enet_RMSE<-min(ENet$results$RMSE)
  Enet_RMSE
  
  #6. 
  coef_Enet=coef(ENet$finalModel,  ENet$bestTune$lambda)
  coef_Enet ### Problema, creo que no está cogiendo mujer
}

{
  # Modelo lineal basico MCO
  model_preuba <- lm(formula = model_form,data = train)
  summary(model_preuba)
  
  stargazer(model_preuba, model_preuba,
            dep.var.labels=c("Ln (Ingreso Hogar)"),
            covariate.labels=c("Personas x Cuarto","N° Ocupados","Mujer",
                               "H (Cotiz. Pension)","H (Reg. Subsidiado)"))
  
}

#### 6. Configuracion del mejor modelo y Est F ----
{
  # 1. 
  tuneGrid_bm <- expand.grid(alpha = 0.05, # between 0 and 1. 
                             lambda = 0.1 ) 
  
  fitControl <- trainControl(method = "cv",
                             number = 10,
                             classProbs = TRUE,
                             verbose=FALSE,
                             savePredictions = T) ##  10 fold CV

  # 2. 
  ENet_bm <- train(model_form,
                   data = train,
                   method = 'glmnet', 
                   trControl = fitControl,
                   tuneGrid = tuneGrid_bm)  #specify the grid 
  # 3. 
  test <- test  %>% mutate(Ln_Ing_Hog_Pred = predict(ENet_bm,
                                                  newdata = test,
                                                        type = "raw"))
  
  # 4. 
  head(test)
  
  # 5.
  test<- test %>% mutate(New_Ing_Hog = exp(Ln_Ing_Hog_Pred))
  test<- test %>% mutate(Pobre_hand = ifelse(New_Ing_Hog < Lp*Npersug, 1, 0))
  test<- test %>% mutate(Pobre_hand_2 = ifelse(Pobre_hand == 1, "Yes", "No"))
  
  head(test, 15)

  # 6.
  confusionMatrix(data = as.factor(test$Pobre_hand_2), 
                  reference = as.factor(test$Pobre), positive="Yes", mode = "prec_recall")
  
  #F1 = 0.4775
}

#### 7. Entrenamiento modelo y prediccion ----
{
  # 1. 
  tuneGrid_bm <- expand.grid(alpha = 0.05, # between 0 and 1. 
                             lambda = 0.1 ) 
  
  fitControl <- trainControl(method = "cv",
                             number = 10,
                             classProbs = TRUE,
                             verbose=FALSE,
                             savePredictions = T) ##  10 fold CV
  
  # 2. 
  # Predicción Kaggle 1
  predictSample <- test_hogares   %>% 
    mutate(Ln_Pobre_test = predict(ENet_bm, newdata = test_hogares, type = "raw"))  %>% 
    mutate (Pobre_test = exp(Ln_Pobre_test),
            Bin_Pobre_test = ifelse(Pobre_test < Lp*Npersug, 1, 0))%>% 
    select(id,Bin_Pobre_test)
  
  predictSample <- predictSample %>% 
    rename(pobre = Bin_Pobre_test) 
  
  write.csv(predictSample,"Med_Indirecta_Linear_EN.csv", row.names = FALSE)
  
}


######### B. Modelo saturado ######################

#### 1. Cargar paquetes ----
{
  install.packages("pacman")
  
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
  
  
  setwd("C:/Users/Juan/Documents/Problem_set_2_BDML/Data")
  
  
  load("base_final.RData")
  colnames(train_hogares) 
}

#### 2. Selección de variables ----
{ #Train
  train_hogares <- train_hogares %>% #seleccionar variables
    dplyr::mutate(Ln_Ing_hog = log(Ingtotugarr),
                  PerXCuarto = Nper/N_cuartos_hog) %>% 
    dplyr::select(-id,
                  -Clase, #ya esta cabecera
                  -P5010, #¿en cuántos de esos cuartos duermen las personas de este hogar?
                  -P5100,#cuando paga por amort (ya esta con ln(cuota)
                  -P5140,#arriendo ya esta con ln,
                  -Ingtotug,
                  -Ingtotugarr,
                  -Li,
                  -Ingpcug,
                  -Ln_Ing_tot_hogar_imp_arr,
                  -Ln_Ing_tot_hogar_per_cap,
                  -Ln_Ing_tot_hogar,
                  -Fex_c,
                  -Depto,
                  -Nper,
                  -Head_Rec_alimento,
                  -Head_Ocupacion,
                  -Head_Rec_vivienda,
                  -Head_EducLevel,
                  -Head_Primas,
                  -N_cuartos_hog,
                  -Cabecera,
                  -Head_Bonificaciones,
                  -Head_Segundo_trabajo,
                  -DormitorXpersona)

  train_hogares <- train_hogares %>%
    mutate(Ln_Ing_hog = ifelse(Ln_Ing_hog == "-Inf", 0, Ln_Ing_hog))
  
  summary(train_hogares$Ln_Ing_hog)
  }

{ # Test
  test_hogares <- test_hogares %>% #seleccionar variables
    dplyr::mutate(PerXCuarto = Nper/N_cuartos_hog) %>% 
    dplyr::select(-Clase, #ya esta cabecera
                  -P5010, #¿en cuántos de esos cuartos duermen las personas de este hogar?
                  -P5100,#cuando paga por amort (ya esta con ln(cuota)
                  -P5140,#arriendo ya esta con ln,
                  -Li,
                  -Fex_c,
                  -Nper,
                  -Depto,
                  -Head_Rec_alimento,
                  -Head_Rec_vivienda,
                  -Head_EducLevel,
                  -Head_Primas,
                  -N_cuartos_hog,
                  -Head_Ocupacion,
                  -Cabecera,
                  -Head_Bonificaciones,
                  -Head_Segundo_trabajo,
                  -DormitorXpersona)
  
  #dejar variables que comparten test y train depsues de crear dummys
  train_hogares <- train_hogares[c(colnames(test_hogares)[2:ncol(test_hogares)],"Ln_Ing_hog", "Pobre")]
}

#### 3. Imbalances ----
{
  view(train_hogares$Ln_Ing_hog)
  
  # Dividir los datos en conjuntos de entrenamiento (train) y prueba (test)
  set.seed(6392) # Para reproducibilidad
  train_indices <- as.integer(createDataPartition(train_hogares$Pobre, p = 0.8, list = FALSE))
  train <- train_hogares[train_indices, ]
  test <- train_hogares[-train_indices, ]
  
  summary(train$Ln_Ing_hog) 
  summary(test$Ln_Ing_hog) 
  
  predictors <- colnames(train  %>% dplyr::select(-Ln_Ing_hog, 
                                                  -Pobre,
                                                  -Lp))
}

#### 4. Configuración del Modelo complejo ----
{ # Modelo 
  model_form <- Ln_Ing_hog ~ as.factor(Dominio) + PerXCuarto + as.factor(Ocup_vivienda) + Fex_dpto + nmujeres + nmenores_5 + nmenores_6_11 + nmenores_12_17 + ntrabajo_menores + nocupados + as.factor(maxEducLevel) + nincapacitados + as.factor(Head_Mujer) + as.factor(Head_ocupado) + as.factor(Head_Afiliado_SS) + as.factor(Head_Reg_subs_salud) + Head_exper_ult_trab + as.factor(Head_Cot_pension) + as.factor(Head_Rec_subsidio) + as.factor(Head_Nivel_formalidad) + as.factor(Head_Oficio) + Ln_Cuota + Ln_Pago_arrien
  
}

#### 5. Elastic Net ----
{ 
  tuneGrid<- expand.grid(alpha= seq(0,1, 0.05), # between 0 and 1. 
                         lambda=seq(0.5, 1.5, 0.5) ) 
  fitControl <- trainControl( 
    method = "cv",
    number = 10) ##  10 fold CV
  
  ENet<-train(model_form,
              data=train_hogares,
              method = 'glmnet', 
              trControl = fitControl,
              tuneGrid = tuneGrid )  #specify the grid 
  
  plot(ENet)
  
  Enet_RMSE<-min(ENet$results$RMSE)
  Enet_RMSE
  ENet$coefnames
}

#### 6. Prueba sobre la submuestra ----
{ # Prueba sobre el testeo
  test <- test  %>% mutate(Ln_Ing_Hog_Pred = predict(ENet,
                                                     newdata = test,
                                                     type = "raw"))
  test<- test %>% mutate(New_Ing_Hog = exp(Ln_Ing_Hog_Pred))
  test<- test %>% mutate(Pobre_hand = ifelse(New_Ing_Hog < Lp*Npersug, 1, 0))
  test<- test %>% mutate(Pobre_hand_2 = ifelse(Pobre_hand == 1, "Yes", "No"))
  
  head(test, 15)
  
  # 6.
  confusionMatrix(data = as.factor(test$Pobre_hand_2), 
                  reference = as.factor(test$Pobre), positive="Yes", mode = "prec_recall")
  
  #F1 = 0.60
}

#### 7. Entrenamiento modelo y predicción ----
{
  # 1. 
  tuneGrid_bm <- expand.grid(alpha = 0.05, # between 0 and 1. 
                             lambda = 0.1 ) 
  
  fitControl <- trainControl(method = "cv",
                             number = 10,
                             classProbs = TRUE,
                             verbose=FALSE,
                             savePredictions = T) ##  10 fold CV
  
  # 2. 
  # Predicción Kaggle 1
  predictSample <- test_hogares   %>% 
    mutate(Ln_Pobre_test = predict(ENet, newdata = test_hogares, type = "raw"))  %>% 
    mutate (Pobre_test = exp(Ln_Pobre_test),
            Bin_Pobre_test = ifelse(Pobre_test < Lp*Npersug, 1, 0))%>% 
    select(id,Bin_Pobre_test)
  
  predictSample <- predictSample %>% 
    rename(pobre = Bin_Pobre_test) 
  
  write.csv(predictSample,"Med_Indirecta_Linear_complejo_EN.csv", row.names = FALSE)
  
}
