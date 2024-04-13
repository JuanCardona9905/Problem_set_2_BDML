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

setwd("C:/Users/user/OneDrive - Universidad de los andes/Big Data y Machine Learning/Problem_set_2_BDML/Data")
load("base_final.RData")
colnames(train_hogares)


#modelo 4 - logit con remuestreo (saturado) SMOTE F1 = 0.66

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
  
  # Sin técnicas de rembalanceo
  ctrl<- trainControl(method = "cv",
                      number = 10,
                      classProbs = TRUE,
                      verbose=FALSE,
                      savePredictions = T)
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
  #F1=0.66
  
  predictSample1 <- test_hogares   %>% 
    mutate(Pobre = predict(pobre_logit_smote, newdata = test_hogares, type = "raw")  
    )  %>% select(id,Pobre)
  
  predictSample1<- predictSample1 %>% 
    mutate(pobre=ifelse(Pobre=="Yes",1,0)) %>% 
    select(id,pobre)
  
  write.csv(predictSample1,"classification_linear_r_smote.csv", row.names = FALSE)
  
  
  
################################################
#predicción indirecta de pobreza#
  
#Idea principal. Realizar un random forest para observar las varibales
# más importantes de la base. De acuerdo con eso tomaré ellas para realizar un
#boosting :D
  train_hogares <- train_hogares %>% #seleccionar variables
    select(-id,
           -Clase, #ya esta cabecera
           -P5010, #¿en cuántos de esos cuartos duermen las personas de este hogar?
           -P5100,#cuando paga por amort (ya esta con ln(cuota)
           -P5140,#arriendo ya esta con ln,
           -Npersug, #no. personas unidad gasto,
           -Li,
           -Fex_c)
  
  test_hogares <- test_hogares %>% #seleccionar variables
    select(-Clase, #ya esta cabecera
           -P5010, #¿en cuántos de esos cuartos duermen las personas de este hogar?
           -P5100,#cuando paga por amort (ya esta con ln(cuota)
           -P5140,#arriendo ya esta con ln
           -Npersug, #no. personas unidad gasto,
           -Fex_c)
  train_hogares<-train_hogares %>% mutate (Lnlp <- log(Lp))
  test_hogares<-test_hogares %>% mutate(Lnlp<- log(Lp))

  
  set.seed(6392) # Para reproducibilidad
  train_indices <- as.integer(createDataPartition(train_hogares$Pobre, p = 0.8, list = FALSE))
  train <- train_hogares[train_indices, ]
  test <- train_hogares[-train_indices, ]
  
  set.seed(2516)
  
  fitControl<-trainControl(method ="cv",
                         number=5)
  
  tree_ranger_grid <- train(
    Ingtotugarr ~ DormitorXpersona + factor(Head_Mujer)  + Dominio + Head_Ocupacion + Head_EducLevel + factor(Head_ocupado) + factor(Head_Rec_subsidio) + Head_Oficio + factor(Head_Segundo_trabajo),
    data = train,
    method = "ranger",
    trControl = fitControl,
    tuneGrid = expand.grid(
      mtry = c(1, 2, 3), 
      splitrule = c("variance", "extratrees", "gini"), 
      min.node.size = c(1, 3, 5)),
    importance="impurity"
  )
  
  tree_ranger_grid
  varImp(tree_ranger_grid)
  test$PredictRFincome <- predict(tree_ranger_grid, newdata = test)
  test$pobre_hat1 <- ifelse(test$PredictRFincome <= test$Lp*test$Nper, 1 ,0)

  test <- test %>% 
    mutate(pobre_hat1=factor(pobre_hat1,levels=c(0,1),labels=c("No","Yes")))
      
      
  confusionMatrix(data = test$pobre_hat1, 
                  reference = test$Pobre, positive="Yes", mode = "prec_recall")

  #F1=0.21
  
  predictSample <- test_hogares   %>% 
    mutate(ingreso_predict= predict(tree_ranger_grid, newdata = test_hogares))  %>% select(id,ingreso_predict,Lp,Nper)
  
  predictSample$pobre <- ifelse(predictSample$ingreso_predict <= predictSample$Lp*predictSample$Nper, 1 ,0)
  predictSample<- predictSample %>% 
    select(id,pobre)
  predictSample1 <- select(predictSample$id, predictSample$pobre) 
write.csv(predictSample,"Regresion_indirecta_random_forest.csv", row.names=FALSE)

            
            