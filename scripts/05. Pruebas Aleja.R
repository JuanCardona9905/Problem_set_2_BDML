########### Modelos de prueba CARTs, Random Forest y Boosting ################

#- 1 | Carga de librerias y base de datos ----------------------------------------------------
rm(list = ls())
require("pacman")
p_load("tidyverse",
       "glmnet",
       "caret",
       "smotefamily",
       "dplyr",
       "dummy",
       "rpart", # Recursive Partition and Regression Trees (To run Trees)
       "rpart.plot", ## for trees graphs
       "Metrics", # Evaluation Metrics for ML
       "MLeval",#*MLeval: Machine Learning Model Evaluation
       "ipred", # For Bagging 
       "pROC",
       "ROSE",#remuestreo ROSE
       "ranger") #random forest 
library("dplyr")
library("tidyverse")

setwd("/Users/aleja/Documents/Maestría Uniandes/Clases/Big Data y Machine Learning/Repositorios Git Hub/Problem_set_2_BDML/Data")
load("base_final.RData")
colnames(train_hogares)
colnames(test_hogares)
colnames(train_personas)
colnames(test_personas)

#- 2 | Modelo 1: Random forest con variables relevantes ---------------------

#Seleccionamos primero las variables más importantes de la base de entrenamiento hogares

train_hogares1 <- dplyr::select(train_hogares, Pobre, Dominio, Depto, N_cuartos_hog, Nper, nmenores_5, nmenores_6_11, 
                                nmenores_12_17, nocupados, nincapacitados, ntrabajo_menores, Head_Mujer, Head_Afiliado_SS, 
                                Head_exper_ult_trab, Head_Rec_alimento, Head_Rec_subsidio, Head_Rec_vivienda, Head_Ocupacion, 
                                Head_Segundo_trabajo, DormitorXpersona, Ln_Cuota, Ln_Pago_arrien, nmujeres, Ocup_vivienda, 
                                Head_Cot_pension)

train_hogares1 <- train_hogares1 %>% 
  mutate(
    Head_Mujer <- factor(Head_Mujer),
    Head_Afiliado_SS <- factor(Head_Afiliado_SS),
    Head_exper_ult_trab <- factor(Head_exper_ult_trab),
    Head_Rec_alimento <- factor(Head_Rec_alimento),
    Head_Rec_subsidio <- factor(Head_Rec_subsidio),
    Head_Rec_vivienda <- factor(Head_Rec_vivienda),
    Head_Ocupacion <- factor(Head_Ocupacion),
    Ocup_vivienda <- factor(Ocup_vivienda),
    Head_Cot_pension <- factor(Head_Cot_pension),
    Head_Segundo_trabajo <- factor(Head_Segundo_trabajo))


#Seleccionamos las mismas variables en la data de test
test_hogares1 <- test_hogares %>% 
  dplyr::select(id,Dominio, Depto, N_cuartos_hog, Nper, nmenores_5, nmenores_6_11, 
         nmenores_12_17, nocupados, nincapacitados, ntrabajo_menores, Head_Mujer, Head_Afiliado_SS, 
         Head_exper_ult_trab, Head_Rec_alimento, Head_Rec_subsidio, Head_Rec_vivienda, Head_Ocupacion, 
         Head_Segundo_trabajo, DormitorXpersona, Ln_Cuota, Ln_Pago_arrien, nmujeres, Ocup_vivienda, 
         Head_Cot_pension) %>% 
  mutate(
    Head_Mujer <- factor(Head_Mujer),
    Head_Afiliado_SS <- factor(Head_Afiliado_SS),
    Head_exper_ult_trab <- factor(Head_exper_ult_trab),
    Head_Rec_alimento <- factor(Head_Rec_alimento),
    Head_Rec_subsidio <- factor(Head_Rec_subsidio),
    Head_Rec_vivienda <- factor(Head_Rec_vivienda),
    Head_Ocupacion <- factor(Head_Ocupacion),
    Head_Segundo_trabajo <- factor(Head_Segundo_trabajo))  


#Vamos a seguir un approach de modelo de clasificación donde 1 es pobre y 0 es no pobre

RF<- ranger(Pobre~Dominio + Depto + N_cuartos_hog + Nper + nmenores_5 + nmenores_6_11 + 
            nmenores_12_17 + nocupados + nincapacitados + ntrabajo_menores + Head_Mujer + Head_Afiliado_SS + 
            Head_exper_ult_trab + Head_Rec_alimento + Head_Rec_subsidio + Head_Rec_vivienda + Head_Ocupacion + 
            Head_Segundo_trabajo + DormitorXpersona + Ln_Cuota + Ln_Pago_arrien + nmujeres + Ocup_vivienda + 
            Head_Cot_pension, 
            data = train_hogares1,
            num.trees= 500, ## Numero de bootstrap samples y arboles a estimar. Default 500  
            mtry= 4,   # N. var aleatoriamente seleccionadas en cada partición. Baggin usa todas las vars.
            min.node.size  = 1, ## Numero minimo de observaciones en un nodo para intentar 
            importance="impurity") 
RF

#Importancia de las variables en random forest

imp<-importance(RF)
imp2<- data.frame(variables= names(imp),
                  importance= imp)

ggplot(imp2, aes(x = reorder(variables, importance) , y =importance )) +
  geom_bar(stat = "identity", fill = "red") +
  labs(title = "Variable ", x = "Importance", y="Variable") +
  theme_minimal() +
  coord_flip() 

# Definamos el proceso de 5-fold cross validation

fiveStats <- function(...) c(twoClassSummary(...), defaultSummary(...))
ctrl<- trainControl(method = "cv",
                    number = 5,
                    summaryFunction = fiveStats,
                    classProbs = TRUE,
                    verbose=FALSE,
                    savePredictions = T)

# Definamos el grid sobre el cual se realizará la búsqueda

mtry_grid<-expand.grid(mtry =c(2,4,6,8), # 8 inclueye bagging
                       min.node.size= c(1, 5, 10, 20, 35, 50), #controla la complejidad del arbol
                       splitrule= 'gini') #splitrule fija en gini. 
mtry_grid

#Usemos train para buscar la mejor combinación de parámetros

cv_RForest <- train(Pobre~Dominio + Depto + N_cuartos_hog + Nper + nmenores_5 + nmenores_6_11 + 
                      nmenores_12_17 + nocupados + nincapacitados + ntrabajo_menores + Head_Mujer + Head_Afiliado_SS + 
                      Head_exper_ult_trab + Head_Rec_alimento + Head_Rec_subsidio + Head_Rec_vivienda + Head_Ocupacion + 
                      Head_Segundo_trabajo + DormitorXpersona + Ln_Cuota + Ln_Pago_arrien + nmujeres + Ocup_vivienda + 
                      Head_Cot_pension, 
                    data = train_hogares1,
                    method = "ranger",
                    trControl = ctrl,
                    metric="ROC",
                    tuneGrid = mtry_grid,
                    ntree=500)


# Veamos el modelo final
cv_RForest$finalModel

# Guardemos el AUC para el test data
rf_pred <- predict(cv_RForest, 
                   newdata = test_hogares1, 
                   type="prob" ## class for class prediction
)

#aucval_rf <- Metrics::auc(pobre,predicted =rf_pred[,2])
#aucval_rf


#Alistemoslo para subirlo a kaggle

predictSample <- test_hogares1   %>% 
  mutate(Pobre = predict(cv_RForest, newdata = test_hogares1, type = "raw"))  %>% dplyr::select(id,Pobre)

predictSample<- predictSample %>% 
  mutate(pobre=ifelse(Pobre=="Yes",1,0)) %>% 
  dplyr::select(id,pobre)

write.csv(predictSample,"random_forest_clasificacion_ale1.csv", row.names = FALSE)

#- 3 | Modelo 2: Random forest con variables relevantes y distintas ---------------------
#WIP
#Seleccionamos primero las variables más importantes de la base de entrenamiento hogares

train_hogares1 <- dplyr::select(train_hogares, Pobre, Dominio, Depto, N_cuartos_hog, Nper, nmenores_5, nmenores_6_11, 
                                nmenores_12_17, nocupados, nincapacitados, ntrabajo_menores, Head_Mujer, Head_Afiliado_SS, 
                                Head_exper_ult_trab, Head_Rec_alimento, Head_Rec_subsidio, Head_Rec_vivienda, Head_Ocupacion, 
                                Head_Segundo_trabajo, DormitorXpersona, Ln_Cuota, Ln_Pago_arrien, nmujeres, Ocup_vivienda, 
                                Head_Cot_pension)

train_hogares1 <- train_hogares1 %>% 
  mutate(
    Head_Mujer <- factor(Head_Mujer),
    Head_Afiliado_SS <- factor(Head_Afiliado_SS),
    Head_exper_ult_trab <- factor(Head_exper_ult_trab),
    Head_Rec_alimento <- factor(Head_Rec_alimento),
    Head_Rec_subsidio <- factor(Head_Rec_subsidio),
    Head_Rec_vivienda <- factor(Head_Rec_vivienda),
    Head_Ocupacion <- factor(Head_Ocupacion),
    Ocup_vivienda <- factor(Ocup_vivienda),
    Head_Cot_pension <- factor(Head_Cot_pension),
    Head_Segundo_trabajo <- factor(Head_Segundo_trabajo))


#Seleccionamos las mismas variables en la data de test
test_hogares1 <- test_hogares %>% 
  dplyr::select(id,Dominio, Depto, N_cuartos_hog, Nper, nmenores_5, nmenores_6_11, 
                nmenores_12_17, nocupados, nincapacitados, ntrabajo_menores, Head_Mujer, Head_Afiliado_SS, 
                Head_exper_ult_trab, Head_Rec_alimento, Head_Rec_subsidio, Head_Rec_vivienda, Head_Ocupacion, 
                Head_Segundo_trabajo, DormitorXpersona, Ln_Cuota, Ln_Pago_arrien, nmujeres, Ocup_vivienda, 
                Head_Cot_pension) %>% 
  mutate(
    Head_Mujer <- factor(Head_Mujer),
    Head_Afiliado_SS <- factor(Head_Afiliado_SS),
    Head_exper_ult_trab <- factor(Head_exper_ult_trab),
    Head_Rec_alimento <- factor(Head_Rec_alimento),
    Head_Rec_subsidio <- factor(Head_Rec_subsidio),
    Head_Rec_vivienda <- factor(Head_Rec_vivienda),
    Head_Ocupacion <- factor(Head_Ocupacion),
    Head_Segundo_trabajo <- factor(Head_Segundo_trabajo))  


#Vamos a seguir un approach de modelo de clasificación donde 1 es pobre y 0 es no pobre

RF<- ranger(Pobre~Dominio + Depto + N_cuartos_hog + Nper + nmenores_5 + nmenores_6_11 + 
              nmenores_12_17 + nocupados + nincapacitados + ntrabajo_menores + Head_Mujer + Head_Afiliado_SS + 
              Head_exper_ult_trab + Head_Rec_alimento + Head_Rec_subsidio + Head_Rec_vivienda + Head_Ocupacion + 
              Head_Segundo_trabajo + DormitorXpersona + Ln_Cuota + Ln_Pago_arrien + nmujeres + Ocup_vivienda + 
              Head_Cot_pension, 
            data = train_hogares1,
            num.trees= 500, ## Numero de bootstrap samples y arboles a estimar. Default 500  
            mtry= 4,   # N. var aleatoriamente seleccionadas en cada partición. Baggin usa todas las vars.
            min.node.size  = 1, ## Numero minimo de observaciones en un nodo para intentar 
            importance="impurity") 
RF

#Importancia de las variables en random forest

imp<-importance(RF)
imp2<- data.frame(variables= names(imp),
                  importance= imp)

ggplot(imp2, aes(x = reorder(variables, importance) , y =importance )) +
  geom_bar(stat = "identity", fill = "red") +
  labs(title = "Variable ", x = "Importance", y="Variable") +
  theme_minimal() +
  coord_flip() 

# Definamos el proceso de 5-fold cross validation

fiveStats <- function(...) c(twoClassSummary(...), defaultSummary(...))
ctrl<- trainControl(method = "cv",
                    number = 5,
                    summaryFunction = fiveStats,
                    classProbs = TRUE,
                    verbose=FALSE,
                    savePredictions = T)

# Definamos el grid sobre el cual se realizará la búsqueda

mtry_grid<-expand.grid(mtry =c(2,4,6,8), # 8 inclueye bagging
                       min.node.size= c(1, 5, 10, 20, 35, 50), #controla la complejidad del arbol
                       splitrule= 'gini') #splitrule fija en gini. 
mtry_grid

#Usemos train para buscar la mejor combinación de parámetros

cv_RForest <- train(Pobre~Dominio + Depto + N_cuartos_hog + Nper + nmenores_5 + nmenores_6_11 + 
                      nmenores_12_17 + nocupados + nincapacitados + ntrabajo_menores + Head_Mujer + Head_Afiliado_SS + 
                      Head_exper_ult_trab + Head_Rec_alimento + Head_Rec_subsidio + Head_Rec_vivienda + Head_Ocupacion + 
                      Head_Segundo_trabajo + DormitorXpersona + Ln_Cuota + Ln_Pago_arrien + nmujeres + Ocup_vivienda + 
                      Head_Cot_pension, 
                    data = train_hogares1,
                    method = "ranger",
                    trControl = ctrl,
                    metric="ROC",
                    tuneGrid = mtry_grid,
                    ntree=500)


# Veamos el modelo final
cv_RForest$finalModel

# Guardemos el AUC para el test data
rf_pred <- predict(cv_RForest, 
                   newdata = test_hogares1, 
                   type="prob" ## class for class prediction
)

#aucval_rf <- Metrics::auc(pobre,predicted =rf_pred[,2])
#aucval_rf


#Alistemoslo para subirlo a kaggle

predictSample <- test_hogares1   %>% 
  mutate(Pobre = predict(cv_RForest, newdata = test_hogares1, type = "raw"))  %>% dplyr::select(id,Pobre)

predictSample<- predictSample %>% 
  mutate(pobre=ifelse(Pobre=="Yes",1,0)) %>% 
  dplyr::select(id,pobre)

write.csv(predictSample,"random_forest_clasificacion_ale1.csv", row.names = FALSE)
