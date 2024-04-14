########### Modelos de prueba de clasificación ###############
########## con métodos CARTs, Random Forest y Boosting ################

#- 1 | Carga de librerias y base de datos ----------------------------------------------------
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
       "DiagrammeR",
       "xgboost",
       "ROSE",#remuestreo ROSE
       "ranger") #random forest 
library("dplyr")


setwd(paste0(wd,"/Data"))

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
    Head_Mujer = factor(Head_Mujer),
    Head_Afiliado_SS = factor(Head_Afiliado_SS),
    Head_exper_ult_trab = factor(Head_exper_ult_trab),
    Head_Rec_alimento = factor(Head_Rec_alimento),
    Head_Rec_subsidio = factor(Head_Rec_subsidio),
    Head_Rec_vivienda = factor(Head_Rec_vivienda),
    Head_Ocupacion = factor(Head_Ocupacion),
    Ocup_vivienda = factor(Ocup_vivienda),
    Head_Cot_pension = factor(Head_Cot_pension),
    Head_Segundo_trabajo = factor(Head_Segundo_trabajo))


#Seleccionamos las mismas variables en la data de test
test_hogares1 <- test_hogares %>% 
  dplyr::select(id,Dominio, Depto, N_cuartos_hog, Nper, nmenores_5, nmenores_6_11, 
                nmenores_12_17, nocupados, nincapacitados, ntrabajo_menores, Head_Mujer, Head_Afiliado_SS, 
                Head_exper_ult_trab, Head_Rec_alimento, Head_Rec_subsidio, Head_Rec_vivienda, Head_Ocupacion, 
                Head_Segundo_trabajo, DormitorXpersona, Ln_Cuota, Ln_Pago_arrien, nmujeres, Ocup_vivienda, 
                Head_Cot_pension) %>% 
  mutate(
    Head_Mujer = factor(Head_Mujer),
    Head_Afiliado_SS = factor(Head_Afiliado_SS),
    Head_exper_ult_trab = factor(Head_exper_ult_trab),
    Head_Rec_alimento = factor(Head_Rec_alimento),
    Head_Rec_subsidio = factor(Head_Rec_subsidio),
    Head_Rec_vivienda = factor(Head_Rec_vivienda),
    Head_Ocupacion = factor(Head_Ocupacion),
    Head_Segundo_trabajo = factor(Head_Segundo_trabajo))  


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

#Puntaje Kagle 0.59
write.csv(predictSample,"random_forest_clasificacion_ale1.csv", row.names = FALSE)

#- 3 | Modelo 2: Random forest con variables relevantes y distintas formas funcionales ---------------------

#Seleccionamos primero las variables más importantes de la base de entrenamiento hogares

train_hogares1 <- dplyr::select(train_hogares, Pobre, Dominio, Depto, N_cuartos_hog, Nper, nmenores_5, nmenores_6_11, 
                                nmenores_12_17, nocupados, nincapacitados, ntrabajo_menores, Head_Mujer, Head_Afiliado_SS, 
                                Head_exper_ult_trab, Head_Rec_alimento, Head_Rec_subsidio, Head_Rec_vivienda, Head_Ocupacion, 
                                Head_Segundo_trabajo, DormitorXpersona, Ln_Cuota, Ln_Pago_arrien, nmujeres, Ocup_vivienda, 
                                Head_Cot_pension)

train_hogares1 <- train_hogares1 %>% 
  mutate(
    Head_Mujer = factor(Head_Mujer),
    Head_Afiliado_SS = factor(Head_Afiliado_SS),
    Head_exper_ult_trab = factor(Head_exper_ult_trab),
    Head_Rec_alimento = factor(Head_Rec_alimento),
    Head_Rec_subsidio = factor(Head_Rec_subsidio),
    Head_Rec_vivienda = factor(Head_Rec_vivienda),
    Head_Ocupacion = factor(Head_Ocupacion),
    Ocup_vivienda = factor(Ocup_vivienda),
    Head_Cot_pension = factor(Head_Cot_pension),
    Head_Segundo_trabajo = factor(Head_Segundo_trabajo))


#Seleccionamos las mismas variables en la data de test
test_hogares1 <- test_hogares %>% 
  dplyr::select(id,Dominio, Depto, N_cuartos_hog, Nper, nmenores_5, nmenores_6_11, 
                nmenores_12_17, nocupados, nincapacitados, ntrabajo_menores, Head_Mujer, Head_Afiliado_SS, 
                Head_exper_ult_trab, Head_Rec_alimento, Head_Rec_subsidio, Head_Rec_vivienda, Head_Ocupacion, 
                Head_Segundo_trabajo, DormitorXpersona, Ln_Cuota, Ln_Pago_arrien, nmujeres, Ocup_vivienda, 
                Head_Cot_pension) 

test_hogares1 <- test_hogares1%>% 
  dplyr::mutate(
    Head_Mujer = factor(Head_Mujer),
    Head_Afiliado_SS = factor(Head_Afiliado_SS),
    Head_exper_ult_trab = factor(Head_exper_ult_trab),
    Head_Rec_alimento = factor(Head_Rec_alimento),
    Head_Rec_subsidio = factor(Head_Rec_subsidio),
    Head_Rec_vivienda = factor(Head_Rec_vivienda),
    Head_Ocupacion = factor(Head_Ocupacion),
    Head_Segundo_trabajo = factor(Head_Segundo_trabajo))  

#Agreguemos algunas interacciones en la data
train_hogares1$nocupados2 <- train_hogares1$nocupados^2
test_hogares1$nocupados2 <- test_hogares1$nocupados^2
train_hogares1$ntrabajo_menores2 <- train_hogares1$ntrabajo_menores^2
test_hogares1$ntrabajo_menores2 <- test_hogares1$ntrabajo_menores^2
train_hogares1$nincapacitados2 <- train_hogares1$nincapacitados^2
test_hogares1$nincapacitados2 <- test_hogares1$nincapacitados^2

#Vamos a seguir un approach de modelo de clasificación donde 1 es pobre y 0 es no pobre

RF<- ranger(Pobre~Dominio + Depto + N_cuartos_hog + Nper + nmenores_5 + nmenores_6_11 + 
              nmenores_12_17 + nocupados + nincapacitados + Head_Mujer +  
              Head_exper_ult_trab + Head_Rec_alimento + Head_Rec_subsidio + Head_Ocupacion + 
              Head_Segundo_trabajo + DormitorXpersona + Ln_Pago_arrien + Ocup_vivienda + 
              Head_Cot_pension + nocupados2 + ntrabajo_menores2 + nincapacitados2, 
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

mtry_grid<-expand.grid(mtry =c(10,15,30), # 8 inclueye bagging
                       min.node.size= c(25,50,100), #controla la complejidad del arbol
                       splitrule= 'gini') #splitrule fija en gini. 
mtry_grid

#Usemos train para buscar la mejor combinación de parámetros

cv_RForest <- train(Pobre~Dominio + Depto + N_cuartos_hog + Nper + nmenores_5 + nmenores_6_11 + 
                      nmenores_12_17 + nocupados + nincapacitados + Head_Mujer +  
                      Head_exper_ult_trab + Head_Rec_alimento + Head_Rec_subsidio + Head_Ocupacion + 
                      Head_Segundo_trabajo + DormitorXpersona + Ln_Pago_arrien + Ocup_vivienda + 
                      Head_Cot_pension + nocupados2 + ntrabajo_menores2 + nincapacitados2, 
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

write.csv(predictSample,"random_forest_clasificacion_ale2.csv", row.names = FALSE)

### Revisemos el F1 del modelo ###

#Primero verifiquemos de la data de entrenamiento cuales variables son distintas a numeric
sapply(train_hogares1, class)

#Cambiamos las variables a numericas
dummys <- dummy(subset(train_hogares1, select = c(Dominio, Depto, Head_Mujer, Head_Afiliado_SS,
                                                  Head_exper_ult_trab,Head_Rec_alimento,Head_Rec_subsidio,
                                                  Head_Rec_vivienda,Head_Ocupacion,Head_Segundo_trabajo,
                                                  Ocup_vivienda,Head_Cot_pension)))
dummys <- as.data.frame(apply(dummys,2,function(x){as.numeric(x)}))

train_hogares1 <- cbind(subset(train_hogares1, select = -c(Dominio, Depto, Head_Mujer, Head_Afiliado_SS,
                                                           Head_exper_ult_trab,Head_Rec_alimento,Head_Rec_subsidio,
                                                           Head_Rec_vivienda,Head_Ocupacion,Head_Segundo_trabajo,
                                                           Ocup_vivienda,Head_Cot_pension)),dummys)

dummys <- dummy(subset(test_hogares1, select = c(Dominio, Depto, Head_Mujer, Head_Afiliado_SS,
                                                 Head_exper_ult_trab,Head_Rec_alimento,Head_Rec_subsidio,
                                                 Head_Rec_vivienda,Head_Ocupacion,Head_Segundo_trabajo,
                                                 Ocup_vivienda,Head_Cot_pension)))
dummys <- as.data.frame(apply(dummys,2,function(x){as.numeric(x)}))

test_hogares1 <- cbind(subset(test_hogares1, select = -c(Dominio, Depto, Head_Mujer, Head_Afiliado_SS,
                                                         Head_exper_ult_trab,Head_Rec_alimento,Head_Rec_subsidio,
                                                         Head_Rec_vivienda,Head_Ocupacion,Head_Segundo_trabajo,
                                                         Ocup_vivienda,Head_Cot_pension)),dummys)

# Dividir los datos en conjuntos de entrenamiento (train) y prueba (test)
set.seed(6392) # Para reproducibilidad
train_indices <- as.integer(createDataPartition(train_hogares1$Pobre, p = 0.8, list = FALSE))
train <- train_hogares1[train_indices, ]
test <- train_hogares1[-train_indices, ]
prop.table(table(train$Pobre))
prop.table(table(test$Pobre))

predictors <- colnames(train  %>% dplyr::select(-Pobre))
smote_output <- SMOTE(X = train[predictors],
                      target = train$Pobre)
smote_data <- smote_output$data

table(train$Pobre)
table(smote_data$class)

model1 <- train(class ~ ., 
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


#F1 = O.6455


#- 4 | Modelo 3: XGBoost predicción del ingreso Ln_Ing_tot_hogar, indirecto de pobreza con interacciones --------------------------

### Arreglo de data ###

#Seleccionamos las variables que necesitamos para predecir el ingreso
train_hogares1 <- dplyr::select(train_hogares, Ln_Ing_tot_hogar, Pobre, Dominio, Depto, P5010, 
                                N_cuartos_hog, Nper, nmenores_5, nmenores_6_11, 
                                nmenores_12_17, nocupados, nincapacitados, ntrabajo_menores, 
                                Head_Mujer, Head_Afiliado_SS, P5140, Npersug, Lp, Li,
                                Head_exper_ult_trab, Head_Rec_alimento, Head_Rec_subsidio, 
                                Head_Rec_vivienda, Head_Ocupacion, maxEducLevel, Head_Primas,
                                Head_Segundo_trabajo, DormitorXpersona, Ln_Cuota, Head_Oficio,
                                Ln_Pago_arrien, nmujeres, Ocup_vivienda, 
                                Head_Cot_pension,Cabecera)

train_hogares1 <- train_hogares1 %>% 
  dplyr::mutate(
    Head_Mujer = factor(Head_Mujer),
    Head_Afiliado_SS = factor(Head_Afiliado_SS),
    Head_exper_ult_trab = factor(Head_exper_ult_trab),
    Head_Rec_alimento = factor(Head_Rec_alimento),
    Head_Rec_subsidio = factor(Head_Rec_subsidio),
    Head_Rec_vivienda = factor(Head_Rec_vivienda),
    Head_Ocupacion = factor(Head_Ocupacion),
    Head_Primas = factor(Head_Primas),
    Head_Segundo_trabajo = factor(Head_Segundo_trabajo),
    Head_Oficio = factor(Head_Oficio),
    Ocup_vivienda = factor(Ocup_vivienda),
    Head_Cot_pension = factor(Head_Cot_pension),
    Head_Segundo_trabajo = factor(Head_Segundo_trabajo),
    Cabecera = factor(Cabecera))


#Seleccionamos las mismas variables en la data de test
test_hogares1 <- test_hogares %>% 
  dplyr::select(id,Dominio, Depto, P5010, 
                N_cuartos_hog, Nper, nmenores_5, nmenores_6_11, 
                nmenores_12_17, nocupados, nincapacitados, ntrabajo_menores, 
                Head_Mujer, Head_Afiliado_SS, P5140, Npersug, Lp, Li,
                Head_exper_ult_trab, Head_Rec_alimento, Head_Rec_subsidio, 
                Head_Rec_vivienda, Head_Ocupacion, maxEducLevel, Head_Primas,
                Head_Segundo_trabajo, DormitorXpersona, Ln_Cuota, Head_Oficio,
                Ln_Pago_arrien, nmujeres, Ocup_vivienda, 
                Head_Cot_pension,Cabecera) %>% 
  mutate(
    Head_Mujer = factor(Head_Mujer),
    Head_Afiliado_SS = factor(Head_Afiliado_SS),
    Head_exper_ult_trab = factor(Head_exper_ult_trab),
    Head_Rec_alimento = factor(Head_Rec_alimento),
    Head_Rec_subsidio = factor(Head_Rec_subsidio),
    Head_Rec_vivienda = factor(Head_Rec_vivienda),
    Head_Ocupacion = factor(Head_Ocupacion),
    Head_Primas = factor(Head_Primas),
    Head_Segundo_trabajo = factor(Head_Segundo_trabajo),
    Head_Oficio = factor(Head_Oficio),
    Ocup_vivienda = factor(Ocup_vivienda),
    Head_Cot_pension = factor(Head_Cot_pension),
    Head_Segundo_trabajo = factor(Head_Segundo_trabajo),
    Cabecera = factor(Cabecera))

## Quitar infinitos ##

#Eliminemos los infinitos en la variable de ln del ingreso

train_hogares1 <- train_hogares1 %>% mutate(Ln_Ing_tot_hogar = ifelse(Ln_Ing_tot_hogar == "-Inf",0,Ln_Ing_tot_hogar)) 

#revisemos que la variable no tenga infinitos

prueba <- train_hogares1 %>% group_by(Ln_Ing_tot_hogar) %>% summarise(n())
view(prueba)
rm(prueba)

#Convirtamos las variables categóricas en numéricas

sapply(train_hogares1,class)

train_hogares1 <- train_hogares1 %>% mutate(Dominio = as.numeric(Dominio),
                                            Depto = as.numeric(Depto),
                                            Head_Mujer = as.numeric(Head_Mujer),
                                            Head_Afiliado_SS = as.numeric(Head_Afiliado_SS),
                                            Head_exper_ult_trab = as.numeric(Head_exper_ult_trab),
                                            Head_Rec_alimento = as.numeric(Head_Rec_alimento),
                                            Head_Rec_subsidio = as.numeric(Head_Rec_subsidio),
                                            Head_Rec_vivienda = as.numeric(Head_Rec_vivienda),
                                            Head_Ocupacion = as.numeric(Head_Ocupacion),
                                            maxEducLevel = as.numeric(maxEducLevel),
                                            Head_Primas = as.numeric(Head_Primas),
                                            Head_Segundo_trabajo = as.numeric(Head_Segundo_trabajo),
                                            Head_Oficio = as.numeric(Head_Oficio),
                                            Ocup_vivienda = as.numeric(Ocup_vivienda),
                                            Head_Cot_pension = as.numeric(Head_Cot_pension),
                                            Cabecera = as.numeric(Cabecera))

test_hogares1 <- test_hogares1 %>% mutate(Dominio = as.numeric(Dominio),
                                          Depto = as.numeric(Depto),
                                          Head_Mujer = as.numeric(Head_Mujer),
                                          Head_Afiliado_SS = as.numeric(Head_Afiliado_SS),
                                          Head_exper_ult_trab = as.numeric(Head_exper_ult_trab),
                                          Head_Rec_alimento = as.numeric(Head_Rec_alimento),
                                          Head_Rec_subsidio = as.numeric(Head_Rec_subsidio),
                                          Head_Rec_vivienda = as.numeric(Head_Rec_vivienda),
                                          Head_Ocupacion = as.numeric(Head_Ocupacion),
                                          maxEducLevel = as.numeric(maxEducLevel),
                                          Head_Primas = as.numeric(Head_Primas),
                                          Head_Segundo_trabajo = as.numeric(Head_Segundo_trabajo),
                                          Head_Oficio = as.numeric(Head_Oficio),
                                          Ocup_vivienda = as.numeric(Ocup_vivienda),
                                          Head_Cot_pension = as.numeric(Head_Cot_pension),
                                          Cabecera = as.numeric(Cabecera))

#Revisemos rápidamente
sapply(train_hogares1,class)
sapply(train_hogares1,class)


# Dividimos la muestra para entrenar al modelo
set.seed(91519) # Importante definir la semilla. 

inTrain <- createDataPartition(
  y = train_hogares1$Ln_Ing_tot_hogar,## La variable dependiente u objetivo 
  p = .7, ## Usamos 70%  de los datos en el conjunto de entrenamiento 
  list = FALSE)


train <- train_hogares1[ inTrain,]
test  <- train_hogares1[-inTrain,]

#Ajuste del modelo

fitControl<-trainControl(method ="cv",
                         number=5)

#Cargamos los parámetros del boosting
grid_xbgoost <- expand.grid(nrounds = c(250),
                            max_depth = c(4),
                            eta = c(0.01), 
                            gamma = c(0), 
                            min_child_weight = c(10, 25),
                            colsample_bytree = c(0.33,0.66), 
                            subsample = c(0.4))
grid_xbgoost


#Entrenamos el modelo
set.seed(91519) # Importante definir la semilla antes entrenar
Xgboost_tree <- train(Ln_Ing_tot_hogar~Dominio + Depto + P5010 + 
                        N_cuartos_hog + Nper + nmenores_5 + nmenores_6_11 + 
                        nmenores_12_17 + nocupados + nincapacitados + ntrabajo_menores + 
                        Head_Mujer + Head_Afiliado_SS + P5140 + Npersug +
                        Head_exper_ult_trab + Head_Rec_alimento + Head_Rec_subsidio + 
                        Head_Rec_vivienda + Head_Ocupacion + maxEducLevel + Head_Primas +
                        Head_Segundo_trabajo + DormitorXpersona + Ln_Cuota + Head_Oficio +
                        Ln_Pago_arrien + nmujeres + Ocup_vivienda + 
                        Head_Cot_pension + Cabecera,
                      data = train, 
                      method = "xgbTree", 
                      trControl = fitControl,
                      tuneGrid=grid_xbgoost
)         

Xgboost_tree


test<- test  %>% mutate(Ln_Ing_tot_hogar_hat=predict(Xgboost_tree,newdata = test))

#Marquemos los que tienen ingreso cero

test <- test %>% mutate(Cero = ifelse(Ln_Ing_tot_hogar==0,1,0))

#Pasemos a exponencial el ingreso

test <- test %>% mutate(Ing_tot_hogar_hat=exp(Ln_Ing_tot_hogar_hat))

# Para medir el F1 primero creemos la variable de pobreza predicha

test <- test %>% mutate(Pobre_hat = ifelse(Ing_tot_hogar_hat<Lp,"Yes","No"))
test$Pobre_hat <- factor(test$Pobre_hat)

#Ahora sí podemos extraer el F1 del confusionmatrix
confusionMatrix(data = test$Pobre_hat, 
                reference = test$Pobre, positive="Yes", mode = "prec_recall")

#F1 = 0.40

#Representemos gráficamente los árboles entrenados
tree_plot <- xgb.plot.tree(model = Xgboost_tree$finalModel,
                           trees = 1:2, plot_width = 1000, plot_height = 500)
tree_plot

## Ahora hagamos la predicción en la data de test hogares

predictSample <- test_hogares1   %>% 
  mutate(Ln_Ing_tot_hogar_hat = predict(Xgboost_tree, newdata = test_hogares1))

#Pasemos a exponencial el ingreso

predictSample <- predictSample %>% mutate(Ing_tot_hogar_hat=exp(Ln_Ing_tot_hogar_hat))

predictSample <- predictSample %>% mutate(Pobre = ifelse(Ing_tot_hogar_hat<Lp,"Yes","No"))
predictSample$Pobre <- factor(predictSample$Pobre)

predictSample <- predictSample %>% dplyr::select(id,Pobre)

predictSample<- predictSample %>% 
  mutate(pobre=ifelse(Pobre=="Yes",1,0)) %>% 
  dplyr::select(id,pobre)

#Kaggle puntaje = 
write.csv(predictSample,"income_prediction_ln_totug_xgboosting_ale2.csv", row.names = FALSE)


#- 5 | Modelo 4: XGBoost predicción del ingreso con Ln_Ingtotugarr --------------------------

### Arreglo de data ###

#Seleccionamos las variables que necesitamos para predecir el ingreso
train_hogares1 <- dplyr::select(train_hogares, Ln_Ing_tot_hogar_imp_arr, Pobre, Dominio, Depto, P5010, 
                                N_cuartos_hog, Nper, nmenores_5, nmenores_6_11, 
                                nmenores_12_17, nocupados, nincapacitados, ntrabajo_menores, 
                                Head_Mujer, Head_Afiliado_SS, P5140, Npersug, Lp, Li,
                                Head_exper_ult_trab, Head_Rec_alimento, Head_Rec_subsidio, 
                                Head_Rec_vivienda, Head_Ocupacion, maxEducLevel, Head_Primas,
                                Head_Segundo_trabajo, DormitorXpersona, Ln_Cuota, Head_Oficio,
                                Ln_Pago_arrien, nmujeres, Ocup_vivienda, 
                                Head_Cot_pension,Cabecera)

train_hogares1 <- train_hogares1 %>% 
  dplyr::mutate(
    Head_Mujer = factor(Head_Mujer),
    Head_Afiliado_SS = factor(Head_Afiliado_SS),
    Head_exper_ult_trab = factor(Head_exper_ult_trab),
    Head_Rec_alimento = factor(Head_Rec_alimento),
    Head_Rec_subsidio = factor(Head_Rec_subsidio),
    Head_Rec_vivienda = factor(Head_Rec_vivienda),
    Head_Ocupacion = factor(Head_Ocupacion),
    Head_Primas = factor(Head_Primas),
    Head_Segundo_trabajo = factor(Head_Segundo_trabajo),
    Head_Oficio = factor(Head_Oficio),
    Ocup_vivienda = factor(Ocup_vivienda),
    Head_Cot_pension = factor(Head_Cot_pension),
    Head_Segundo_trabajo = factor(Head_Segundo_trabajo),
    Cabecera = factor(Cabecera))


#Seleccionamos las mismas variables en la data de test
test_hogares1 <- test_hogares %>% 
  dplyr::select(id,Dominio, Depto, P5010, 
                N_cuartos_hog, Nper, nmenores_5, nmenores_6_11, 
                nmenores_12_17, nocupados, nincapacitados, ntrabajo_menores, 
                Head_Mujer, Head_Afiliado_SS, P5140, Npersug, Lp, Li,
                Head_exper_ult_trab, Head_Rec_alimento, Head_Rec_subsidio, 
                Head_Rec_vivienda, Head_Ocupacion, maxEducLevel, Head_Primas,
                Head_Segundo_trabajo, DormitorXpersona, Ln_Cuota, Head_Oficio,
                Ln_Pago_arrien, nmujeres, Ocup_vivienda, 
                Head_Cot_pension,Cabecera) %>% 
  mutate(
    Head_Mujer = factor(Head_Mujer),
    Head_Afiliado_SS = factor(Head_Afiliado_SS),
    Head_exper_ult_trab = factor(Head_exper_ult_trab),
    Head_Rec_alimento = factor(Head_Rec_alimento),
    Head_Rec_subsidio = factor(Head_Rec_subsidio),
    Head_Rec_vivienda = factor(Head_Rec_vivienda),
    Head_Ocupacion = factor(Head_Ocupacion),
    Head_Primas = factor(Head_Primas),
    Head_Segundo_trabajo = factor(Head_Segundo_trabajo),
    Head_Oficio = factor(Head_Oficio),
    Ocup_vivienda = factor(Ocup_vivienda),
    Head_Cot_pension = factor(Head_Cot_pension),
    Head_Segundo_trabajo = factor(Head_Segundo_trabajo),
    Cabecera = factor(Cabecera))

## Quitar infinitos ##

#Eliminemos los infinitos en la variable de ln del ingreso

train_hogares1 <- train_hogares1 %>% mutate(Ln_Ing_tot_hogar_imp_arr = ifelse(Ln_Ing_tot_hogar_imp_arr == "-Inf",0,Ln_Ing_tot_hogar_imp_arr)) 

#revisemos que la variable no tenga infinitos

prueba <- train_hogares1 %>% group_by(Ln_Ing_tot_hogar_imp_arr) %>% summarise(n())
rm(prueba)

#Convirtamos las variables categóricas en numéricas

sapply(train_hogares1,class)

train_hogares1 <- train_hogares1 %>% mutate(Dominio = as.numeric(Dominio),
                                            Depto = as.numeric(Depto),
                                            Head_Mujer = as.numeric(Head_Mujer),
                                            Head_Afiliado_SS = as.numeric(Head_Afiliado_SS),
                                            Head_exper_ult_trab = as.numeric(Head_exper_ult_trab),
                                            Head_Rec_alimento = as.numeric(Head_Rec_alimento),
                                            Head_Rec_subsidio = as.numeric(Head_Rec_subsidio),
                                            Head_Rec_vivienda = as.numeric(Head_Rec_vivienda),
                                            Head_Ocupacion = as.numeric(Head_Ocupacion),
                                            maxEducLevel = as.numeric(maxEducLevel),
                                            Head_Primas = as.numeric(Head_Primas),
                                            Head_Segundo_trabajo = as.numeric(Head_Segundo_trabajo),
                                            Head_Oficio = as.numeric(Head_Oficio),
                                            Ocup_vivienda = as.numeric(Ocup_vivienda),
                                            Head_Cot_pension = as.numeric(Head_Cot_pension),
                                            Cabecera = as.numeric(Cabecera))

test_hogares1 <- test_hogares1 %>% mutate(Dominio = as.numeric(Dominio),
                                          Depto = as.numeric(Depto),
                                          Head_Mujer = as.numeric(Head_Mujer),
                                          Head_Afiliado_SS = as.numeric(Head_Afiliado_SS),
                                          Head_exper_ult_trab = as.numeric(Head_exper_ult_trab),
                                          Head_Rec_alimento = as.numeric(Head_Rec_alimento),
                                          Head_Rec_subsidio = as.numeric(Head_Rec_subsidio),
                                          Head_Rec_vivienda = as.numeric(Head_Rec_vivienda),
                                          Head_Ocupacion = as.numeric(Head_Ocupacion),
                                          maxEducLevel = as.numeric(maxEducLevel),
                                          Head_Primas = as.numeric(Head_Primas),
                                          Head_Segundo_trabajo = as.numeric(Head_Segundo_trabajo),
                                          Head_Oficio = as.numeric(Head_Oficio),
                                          Ocup_vivienda = as.numeric(Ocup_vivienda),
                                          Head_Cot_pension = as.numeric(Head_Cot_pension),
                                          Cabecera = as.numeric(Cabecera))

#Revisemos rápidamente
sapply(train_hogares1,class)
sapply(train_hogares1,class)


# Dividimos la muestra para entrenar al modelo
set.seed(91519) # Importante definir la semilla. 

inTrain <- createDataPartition(
  y = train_hogares1$Ln_Ing_tot_hogar_imp_arr,## La variable dependiente u objetivo 
  p = .7, ## Usamos 70%  de los datos en el conjunto de entrenamiento 
  list = FALSE)


train <- train_hogares1[ inTrain,]
test  <- train_hogares1[-inTrain,]

#Ajuste del modelo

fitControl<-trainControl(method ="cv",
                         number=5)

#Cargamos los parámetros del boosting
grid_xbgoost <- expand.grid(nrounds = c(250),
                            max_depth = c(4),
                            eta = c(0.01), 
                            gamma = c(0), 
                            min_child_weight = c(10, 25),
                            colsample_bytree = c(0.33,0.66), 
                            subsample = c(0.4))
grid_xbgoost


#Entrenamos el modelo
set.seed(91519) # Importante definir la semilla antes entrenar
Xgboost_tree <- train(Ln_Ing_tot_hogar_imp_arr~Dominio + Depto + P5010 + P5010^2 + 
                        N_cuartos_hog + Nper + nmenores_5 + nmenores_6_11 + 
                        nmenores_12_17 + nocupados + nincapacitados + ntrabajo_menores + 
                        Head_Mujer + Head_Afiliado_SS + Npersug + Npersug^2 +
                        Head_Rec_subsidio + Head_Rec_vivienda + maxEducLevel + 
                        Head_Segundo_trabajo + DormitorXpersona^2 + Ln_Cuota + Head_Oficio +
                        Ln_Pago_arrien + nmujeres^2 + Ocup_vivienda + 
                        Head_Cot_pension + Cabecera,
                      data = train, 
                      method = "xgbTree", 
                      trControl = fitControl,
                      tuneGrid=grid_xbgoost
)         

Xgboost_tree


test<- test  %>% mutate(Ln_Ing_tot_hogar_imp_arr_hat=predict(Xgboost_tree,newdata = test))

#Marquemos los que tienen ingreso cero

test <- test %>% mutate(Cero = ifelse(Ln_Ing_tot_hogar_imp_arr==0,1,0))

#Pasemos a exponencial el ingreso

test <- test %>% mutate(Ing_tot_hogar_imp_arr_hat=exp(Ln_Ing_tot_hogar_imp_arr))

# Para medir el F1 primero creemos la variable de pobreza predicha

test <- test %>% mutate(Pobre_hat = ifelse(Ing_tot_hogar_imp_arr_hat<Lp*Npersug,"Yes","No"))
test$Pobre_hat <- factor(test$Pobre_hat)

#Ahora sí podemos extraer el F1 del confusionmatrix
confusionMatrix(data = test$Pobre_hat, 
                reference = test$Pobre, positive="Yes", mode = "prec_recall")

#F1 = 1.0

#Representemos gráficamente los árboles entrenados
tree_plot <- xgb.plot.tree(model = Xgboost_tree$finalModel,
                           trees = 1:2, plot_width = 1000, plot_height = 500)
tree_plot

## Ahora hagamos la predicción en la data de test hogares

predictSample <- test_hogares1   %>% 
  mutate(Ln_Ing_tot_hogar_imp_arr_hat = predict(Xgboost_tree, newdata = test_hogares1))

#Pasemos a exponencial el ingreso

predictSample <- predictSample %>% mutate(Ing_tot_hogar_imp_arr_hat=exp(Ln_Ing_tot_hogar_imp_arr_hat))

predictSample <- predictSample %>% mutate(Pobre = ifelse(Ing_tot_hogar_imp_arr_hat<Lp*Npersug,"Yes","No"))
predictSample$Pobre <- factor(predictSample$Pobre)

predictSample <- predictSample %>% dplyr::select(id,Pobre)

predictSample<- predictSample %>% 
  mutate(pobre=ifelse(Pobre=="Yes",1,0)) %>% 
  dplyr::select(id,pobre)

#Kaggle puntaje = 
write.csv(predictSample,"income_prediction_ln_Ingtotugarr_xgboosting_ale4.csv", row.names = FALSE)


#- 6 | Modelo 5: XGBoost predicción de pobreza con variables e interacciones --------------------------

### Arreglo de data ###

#Seleccionamos las variables que necesitamos para predecir el ingreso
train_hogares1 <- dplyr::select(train_hogares, Pobre, Dominio, Depto, P5010, 
                                N_cuartos_hog, Nper, nmenores_5, nmenores_6_11, 
                                nmenores_12_17, nocupados, nincapacitados, ntrabajo_menores, 
                                Head_Mujer, Head_Afiliado_SS, P5140, Npersug, Lp, Li,
                                Head_exper_ult_trab, Head_Rec_alimento, Head_Rec_subsidio, 
                                Head_Rec_vivienda, Head_Ocupacion, maxEducLevel, Head_Primas,
                                Head_Segundo_trabajo, DormitorXpersona, Ln_Cuota, Head_Oficio,
                                Ln_Pago_arrien, nmujeres, Ocup_vivienda, 
                                Head_Cot_pension,Cabecera)

train_hogares1 <- train_hogares1 %>% 
  dplyr::mutate(
    Head_Mujer = factor(Head_Mujer),
    Head_Afiliado_SS = factor(Head_Afiliado_SS),
    Head_exper_ult_trab = factor(Head_exper_ult_trab),
    Head_Rec_alimento = factor(Head_Rec_alimento),
    Head_Rec_subsidio = factor(Head_Rec_subsidio),
    Head_Rec_vivienda = factor(Head_Rec_vivienda),
    Head_Ocupacion = factor(Head_Ocupacion),
    Head_Primas = factor(Head_Primas),
    Head_Segundo_trabajo = factor(Head_Segundo_trabajo),
    Head_Oficio = factor(Head_Oficio),
    Ocup_vivienda = factor(Ocup_vivienda),
    Head_Cot_pension = factor(Head_Cot_pension),
    Head_Segundo_trabajo = factor(Head_Segundo_trabajo),
    Cabecera = factor(Cabecera))


#Seleccionamos las mismas variables en la data de test
test_hogares1 <- test_hogares %>% 
  dplyr::select(id,Dominio, Depto, P5010, 
                N_cuartos_hog, Nper, nmenores_5, nmenores_6_11, 
                nmenores_12_17, nocupados, nincapacitados, ntrabajo_menores, 
                Head_Mujer, Head_Afiliado_SS, P5140, Npersug, Lp, Li,
                Head_exper_ult_trab, Head_Rec_alimento, Head_Rec_subsidio, 
                Head_Rec_vivienda, Head_Ocupacion, maxEducLevel, Head_Primas,
                Head_Segundo_trabajo, DormitorXpersona, Ln_Cuota, Head_Oficio,
                Ln_Pago_arrien, nmujeres, Ocup_vivienda, 
                Head_Cot_pension,Cabecera) %>% 
  mutate(
    Head_Mujer = factor(Head_Mujer),
    Head_Afiliado_SS = factor(Head_Afiliado_SS),
    Head_exper_ult_trab = factor(Head_exper_ult_trab),
    Head_Rec_alimento = factor(Head_Rec_alimento),
    Head_Rec_subsidio = factor(Head_Rec_subsidio),
    Head_Rec_vivienda = factor(Head_Rec_vivienda),
    Head_Ocupacion = factor(Head_Ocupacion),
    Head_Primas = factor(Head_Primas),
    Head_Segundo_trabajo = factor(Head_Segundo_trabajo),
    Head_Oficio = factor(Head_Oficio),
    Ocup_vivienda = factor(Ocup_vivienda),
    Head_Cot_pension = factor(Head_Cot_pension),
    Head_Segundo_trabajo = factor(Head_Segundo_trabajo),
    Cabecera = factor(Cabecera))


#Convirtamos las variables categóricas en numéricas

sapply(train_hogares1,class)

train_hogares1 <- train_hogares1 %>% mutate(Dominio = as.numeric(Dominio),
                                            Depto = as.numeric(Depto),
                                            Head_Mujer = as.numeric(Head_Mujer),
                                            Head_Afiliado_SS = as.numeric(Head_Afiliado_SS),
                                            Head_exper_ult_trab = as.numeric(Head_exper_ult_trab),
                                            Head_Rec_alimento = as.numeric(Head_Rec_alimento),
                                            Head_Rec_subsidio = as.numeric(Head_Rec_subsidio),
                                            Head_Rec_vivienda = as.numeric(Head_Rec_vivienda),
                                            Head_Ocupacion = as.numeric(Head_Ocupacion),
                                            maxEducLevel = as.numeric(maxEducLevel),
                                            Head_Primas = as.numeric(Head_Primas),
                                            Head_Segundo_trabajo = as.numeric(Head_Segundo_trabajo),
                                            Head_Oficio = as.numeric(Head_Oficio),
                                            Ocup_vivienda = as.numeric(Ocup_vivienda),
                                            Head_Cot_pension = as.numeric(Head_Cot_pension),
                                            Cabecera = as.numeric(Cabecera))

test_hogares1 <- test_hogares1 %>% mutate(Dominio = as.numeric(Dominio),
                                          Depto = as.numeric(Depto),
                                          Head_Mujer = as.numeric(Head_Mujer),
                                          Head_Afiliado_SS = as.numeric(Head_Afiliado_SS),
                                          Head_exper_ult_trab = as.numeric(Head_exper_ult_trab),
                                          Head_Rec_alimento = as.numeric(Head_Rec_alimento),
                                          Head_Rec_subsidio = as.numeric(Head_Rec_subsidio),
                                          Head_Rec_vivienda = as.numeric(Head_Rec_vivienda),
                                          Head_Ocupacion = as.numeric(Head_Ocupacion),
                                          maxEducLevel = as.numeric(maxEducLevel),
                                          Head_Primas = as.numeric(Head_Primas),
                                          Head_Segundo_trabajo = as.numeric(Head_Segundo_trabajo),
                                          Head_Oficio = as.numeric(Head_Oficio),
                                          Ocup_vivienda = as.numeric(Ocup_vivienda),
                                          Head_Cot_pension = as.numeric(Head_Cot_pension),
                                          Cabecera = as.numeric(Cabecera))

#Revisemos rápidamente
sapply(train_hogares1,class)
sapply(train_hogares1,class)


# Dividimos la muestra para entrenar al modelo
set.seed(91519) # Importante definir la semilla. 

inTrain <- createDataPartition(
  y = train_hogares1$Pobre,## La variable dependiente u objetivo 
  p = .8, ## Usamos 70%  de los datos en el conjunto de entrenamiento 
  list = FALSE)


train <- train_hogares1[ inTrain,]
test  <- train_hogares1[-inTrain,]

#Ajuste del modelo

fiveStats <- function(...) c(twoClassSummary(...), defaultSummary(...))  ## Para usar ROC) (u otras más) para tuning

ctrl<- trainControl(method = "cv",
                    number = 5,
                    summaryFunction = fiveStats,
                    classProbs = TRUE, 
                    verbose=FALSE,
                    savePredictions = T)

#Cargamos los parámetros del boosting
grid_xbgoost <- expand.grid(nrounds = c(250,500),
                            max_depth = c(1, 2),
                            eta = c(0.1,  0.01), 
                            gamma = c(0, 1), 
                            min_child_weight = c(10, 25),
                            colsample_bytree = c(0.4, 0.7), 
                            subsample = c(0.7))
grid_xbgoost


#Entrenamos el modelo
set.seed(91519) # Importante definir la semilla antes entrenar
Xgboost_tree <- train(Pobre~Dominio + Depto + P5010 + P5010^2 + 
                        N_cuartos_hog + Nper + nmenores_5 + nmenores_6_11 + 
                        nmenores_12_17 + nocupados + nincapacitados + ntrabajo_menores + 
                        Head_Mujer + Head_Afiliado_SS + Npersug + Npersug^2 +
                        Head_Rec_subsidio + Head_Rec_vivienda + maxEducLevel + 
                        Head_Segundo_trabajo + DormitorXpersona^2 + Ln_Cuota + Head_Oficio +
                        Ln_Pago_arrien + nmujeres^2 + Ocup_vivienda + 
                        Head_Cot_pension + Cabecera,
                      data = train, 
                      method = "xgbTree", 
                      trControl = ctrl,
                      tuneGrid=grid_xbgoost,
                      metric = "ROC",
                      verbosity = 0
)         


Xgboost_tree


test<- test  %>% mutate(Pobre_hat=predict(Xgboost_tree,newdata = test,
                                          type = "raw"))


confusionMatrix(data = test$Pobre_hat, 
                reference = test$Pobre, positive="Yes", mode = "prec_recall")

#F1 = 1.0

#Representemos gráficamente los árboles entrenados
tree_plot <- xgb.plot.tree(model = Xgboost_tree$finalModel,
                           trees = 1:2, plot_width = 1000, plot_height = 500)
tree_plot

## Ahora hagamos la predicción en la data de test hogares

predictSample <- test_hogares1   %>% 
  mutate(Pobre = predict(Xgboost_tree, newdata = test_hogares1, type = "raw")) %>% dplyr::select(id,Pobre)


predictSample<- predictSample %>% 
  mutate(pobre=ifelse(Pobre=="Yes",1,0)) %>% 
  dplyr::select(id,pobre)

#Kaggle puntaje = 
write.csv(predictSample,"classification_xgboosting_ale5.csv", row.names = FALSE)

#- 7 | Modelo 6: XGBoost predicción de pobreza con todas las variables e interacciones --------------------------

### Arreglo de data ###

#Seleccionamos las variables que necesitamos para predecir el ingreso
train_hogares1 <- train_hogares

#Seleccionamos las mismas variables en la data de test
test_hogares1 <- test_hogares 


#Convirtamos las variables categóricas en numéricas

sapply(train_hogares1,class)

train_hogares1 <- train_hogares1 %>% mutate(Dominio = as.numeric(Dominio),
                                            Depto = as.numeric(Depto),
                                            Head_EducLevel = as.numeric(Head_EducLevel),
                                            Head_Ocupacion = as.numeric(Head_Ocupacion),
                                            maxEducLevel = as.numeric(maxEducLevel),
                                            Head_Oficio = as.numeric(Head_Oficio),
                                            Ocup_vivienda = as.numeric(Ocup_vivienda))

test_hogares1 <- test_hogares1 %>% mutate(Dominio = as.numeric(Dominio),
                                          Depto = as.numeric(Depto),
                                          Head_EducLevel = as.numeric(Head_EducLevel),
                                          Head_Ocupacion = as.numeric(Head_Ocupacion),
                                          maxEducLevel = as.numeric(maxEducLevel),
                                          Head_Oficio = as.numeric(Head_Oficio),
                                          Ocup_vivienda = as.numeric(Ocup_vivienda))

#Revisemos rápidamente
sapply(train_hogares1,class)
sapply(train_hogares1,class)


# Dividimos la muestra para entrenar al modelo
set.seed(91519) # Importante definir la semilla. 

inTrain <- createDataPartition(
  y = train_hogares1$Pobre,## La variable dependiente u objetivo 
  p = .8, ## Usamos 70%  de los datos en el conjunto de entrenamiento 
  list = FALSE)


train <- train_hogares1[ inTrain,]
test  <- train_hogares1[-inTrain,]

#Ajuste del modelo

fiveStats <- function(...) c(twoClassSummary(...), defaultSummary(...))  ## Para usar ROC) (u otras más) para tuning

ctrl<- trainControl(method = "cv",
                    number = 5,
                    summaryFunction = fiveStats,
                    classProbs = TRUE, 
                    verbose=FALSE,
                    savePredictions = T)

#Cargamos los parámetros del boosting
grid_xbgoost <- expand.grid(nrounds = c(250,500),
                            max_depth = c(1, 2),
                            eta = c(0.1,  0.01), 
                            gamma = c(0, 1), 
                            min_child_weight = c(10, 25),
                            colsample_bytree = c(0.4, 0.7), 
                            subsample = c(0.7))
grid_xbgoost


#Entrenamos el modelo
set.seed(91519) # Importante definir la semilla antes entrenar
Xgboost_tree <- train(Pobre~Dominio + Depto + P5010 + P5010^2 + 
                        N_cuartos_hog + Nper + nmenores_5 + nmenores_6_11 + 
                        nmenores_12_17 + nocupados + nincapacitados + ntrabajo_menores + 
                        Head_Mujer + Head_Afiliado_SS + Npersug + Npersug^2 +
                        Head_Rec_subsidio + Head_Rec_vivienda + maxEducLevel + 
                        Head_Segundo_trabajo + DormitorXpersona + DormitorXpersona^2 + Ln_Cuota + Head_Oficio +
                        Ln_Pago_arrien + nmujeres + nmujeres^2 + Ocup_vivienda + 
                        Head_Cot_pension + Cabecera + Clase + P5100 + P5140 + Fex_c + Fex_dpto +
                        Head_EducLevel + Head_ocupado + Head_Reg_subs_salud + Head_exper_ult_trab + Head_Rec_alimento +
                        Head_Nivel_formalidad + Head_Ocupacion + Head_Primas + Head_Bonificaciones,
                      data = train, 
                      method = "xgbTree", 
                      trControl = ctrl,
                      tuneGrid=grid_xbgoost,
                      metric = "ROC",
                      verbosity = 0
)         




Xgboost_tree


test<- test  %>% mutate(Pobre_hat=predict(Xgboost_tree,newdata = test,
                                          type = "raw"))


confusionMatrix(data = test$Pobre_hat, 
                reference = test$Pobre, positive="Yes", mode = "prec_recall")

#F1 = 0.65

#Representemos gráficamente los árboles entrenados
tree_plot <- xgb.plot.tree(model = Xgboost_tree$finalModel,
                           trees = 1:2, plot_width = 1000, plot_height = 500)
tree_plot

## Ahora hagamos la predicción en la data de test hogares

predictSample <- test_hogares1   %>% 
  mutate(Pobre = predict(Xgboost_tree, newdata = test_hogares1, type = "raw")) %>% dplyr::select(id,Pobre)


predictSample<- predictSample %>% 
  mutate(pobre=ifelse(Pobre=="Yes",1,0)) %>% 
  dplyr::select(id,pobre)

#Kaggle puntaje = 
write.csv(predictSample,"classification_xgboosting_ale6.csv", row.names = FALSE)



########### Modelos de prueba de predicción indirecta ##############
########### con métodos CARTs, Random Forest y Boosting ################

#- 1 | Carga de librerias y base de datos ----------------------------------------------------
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
       "xgboost",
       "ROSE",#remuestreo ROSE
       "ranger") #random forest 
library("dplyr")

load("base_final.RData")
colnames(train_hogares)
colnames(test_hogares)

#- 2 | Pruebas varias ------------------------------------------------

#Quiero saber cuál variable de ingreso es mejor tomar para hacer las predicciones

#Primero creemos unas variables provisionales donde podamos ver si existe
#alguna que aunque sea menor que la linea de pobreza, aparezca como no pobre

train_hogares <- train_hogares %>% mutate(test_ingtotug = ifelse(Ingtotug<Lp,1,0),
                                          test_ingtotugarr = ifelse(Ingtotugarr<Lp,1,0),
                                          test_ingpcup = ifelse(Ingpcug<Lp,1,0),)

prueba <- train_hogares %>% 
  group_by(Ingtotug,Ingtotugarr,Ingpcug, Lp, Pobre, 
           test_ingtotug, test_ingtotugarr, test_ingpcup) %>% 
  summarise(n())

view(prueba)

#- 3 | Modelo 1: XGBoost con Ingtotug con variables relevantes ---------------------

#Seleccionamos las variables que necesitamos para predecir el ingreso
train_hogares1 <- dplyr::select(train_hogares, Ingtotug,
                                Pobre, Dominio, Depto, P5010, 
                                N_cuartos_hog, Nper, nmenores_5, nmenores_6_11, 
                                nmenores_12_17, nocupados, nincapacitados, ntrabajo_menores, 
                                Head_Mujer, Head_Afiliado_SS, P5140, Npersug, Lp, Li,
                                Head_exper_ult_trab, Head_Rec_alimento, Head_Rec_subsidio, 
                                Head_Rec_vivienda, Head_Ocupacion, maxEducLevel, Head_Primas,
                                Head_Segundo_trabajo, DormitorXpersona, Ln_Cuota, Head_Oficio,
                                Ln_Pago_arrien, nmujeres, Ocup_vivienda, 
                                Head_Cot_pension,Cabecera)

train_hogares1 <- train_hogares1 %>% 
  dplyr::mutate(
    Head_Mujer = factor(Head_Mujer),
    Head_Afiliado_SS = factor(Head_Afiliado_SS),
    Head_exper_ult_trab = factor(Head_exper_ult_trab),
    Head_Rec_alimento = factor(Head_Rec_alimento),
    Head_Rec_subsidio = factor(Head_Rec_subsidio),
    Head_Rec_vivienda = factor(Head_Rec_vivienda),
    Head_Ocupacion = factor(Head_Ocupacion),
    Head_Primas = factor(Head_Primas),
    Head_Segundo_trabajo = factor(Head_Segundo_trabajo),
    Head_Oficio = factor(Head_Oficio),
    Ocup_vivienda = factor(Ocup_vivienda),
    Head_Cot_pension = factor(Head_Cot_pension),
    Head_Segundo_trabajo = factor(Head_Segundo_trabajo),
    Cabecera = factor(Cabecera))


#Seleccionamos las mismas variables en la data de test
test_hogares1 <- test_hogares %>% 
  dplyr::select(id,Dominio, Depto, P5010, 
                N_cuartos_hog, Nper, nmenores_5, nmenores_6_11, 
                nmenores_12_17, nocupados, nincapacitados, ntrabajo_menores, 
                Head_Mujer, Head_Afiliado_SS, P5140, Npersug, Lp, Li,
                Head_exper_ult_trab, Head_Rec_alimento, Head_Rec_subsidio, 
                Head_Rec_vivienda, Head_Ocupacion, maxEducLevel, Head_Primas,
                Head_Segundo_trabajo, DormitorXpersona, Ln_Cuota, Head_Oficio,
                Ln_Pago_arrien, nmujeres, Ocup_vivienda, 
                Head_Cot_pension,Cabecera) %>% 
  mutate(
    Head_Mujer = factor(Head_Mujer),
    Head_Afiliado_SS = factor(Head_Afiliado_SS),
    Head_exper_ult_trab = factor(Head_exper_ult_trab),
    Head_Rec_alimento = factor(Head_Rec_alimento),
    Head_Rec_subsidio = factor(Head_Rec_subsidio),
    Head_Rec_vivienda = factor(Head_Rec_vivienda),
    Head_Ocupacion = factor(Head_Ocupacion),
    Head_Primas = factor(Head_Primas),
    Head_Segundo_trabajo = factor(Head_Segundo_trabajo),
    Head_Oficio = factor(Head_Oficio),
    Ocup_vivienda = factor(Ocup_vivienda),
    Head_Cot_pension = factor(Head_Cot_pension),
    Head_Segundo_trabajo = factor(Head_Segundo_trabajo),
    Cabecera = factor(Cabecera))


#Convirtamos las variables categóricas en numéricas

sapply(train_hogares1,class)

train_hogares1 <- train_hogares1 %>% mutate(Dominio = as.numeric(Dominio),
                                            Depto = as.numeric(Depto),
                                            Head_Mujer = as.numeric(Head_Mujer),
                                            Head_Afiliado_SS = as.numeric(Head_Afiliado_SS),
                                            Head_exper_ult_trab = as.numeric(Head_exper_ult_trab),
                                            Head_Rec_alimento = as.numeric(Head_Rec_alimento),
                                            Head_Rec_subsidio = as.numeric(Head_Rec_subsidio),
                                            Head_Rec_vivienda = as.numeric(Head_Rec_vivienda),
                                            Head_Ocupacion = as.numeric(Head_Ocupacion),
                                            maxEducLevel = as.numeric(maxEducLevel),
                                            Head_Primas = as.numeric(Head_Primas),
                                            Head_Segundo_trabajo = as.numeric(Head_Segundo_trabajo),
                                            Head_Oficio = as.numeric(Head_Oficio),
                                            Ocup_vivienda = as.numeric(Ocup_vivienda),
                                            Head_Cot_pension = as.numeric(Head_Cot_pension),
                                            Cabecera = as.numeric(Cabecera))

test_hogares1 <- test_hogares1 %>% mutate(Dominio = as.numeric(Dominio),
                                          Depto = as.numeric(Depto),
                                          Head_Mujer = as.numeric(Head_Mujer),
                                          Head_Afiliado_SS = as.numeric(Head_Afiliado_SS),
                                          Head_exper_ult_trab = as.numeric(Head_exper_ult_trab),
                                          Head_Rec_alimento = as.numeric(Head_Rec_alimento),
                                          Head_Rec_subsidio = as.numeric(Head_Rec_subsidio),
                                          Head_Rec_vivienda = as.numeric(Head_Rec_vivienda),
                                          Head_Ocupacion = as.numeric(Head_Ocupacion),
                                          maxEducLevel = as.numeric(maxEducLevel),
                                          Head_Primas = as.numeric(Head_Primas),
                                          Head_Segundo_trabajo = as.numeric(Head_Segundo_trabajo),
                                          Head_Oficio = as.numeric(Head_Oficio),
                                          Ocup_vivienda = as.numeric(Ocup_vivienda),
                                          Head_Cot_pension = as.numeric(Head_Cot_pension),
                                          Cabecera = as.numeric(Cabecera))

#Revisemos rápidamente
sapply(train_hogares1,class)
sapply(train_hogares1,class)


# Dividimos la muestra para entrenar al modelo
set.seed(91519) # Importante definir la semilla. 

inTrain <- createDataPartition(
  y = train_hogares1$Ingtotug,## La variable dependiente u objetivo 
  p = .7, ## Usamos 70%  de los datos en el conjunto de entrenamiento 
  list = FALSE)


train <- train_hogares1[ inTrain,]
test  <- train_hogares1[-inTrain,]

#Ajuste del modelo

fitControl<-trainControl(method ="cv",
                         number=3)

#Cargamos los parámetros del boosting
grid_xbgoost <- expand.grid(nrounds = c(250),
                            max_depth = c(4),
                            eta = c(0.01), 
                            gamma = c(0), 
                            min_child_weight = c(10, 25),
                            colsample_bytree = c(0.33,0.66), 
                            subsample = c(0.4))
grid_xbgoost


#Entrenamos el modelo
set.seed(91519) # Importante definir la semilla antes entrenar
Xgboost_tree <- train(Ingtotug~Dominio + Depto + P5010 + 
                        N_cuartos_hog + Nper + nmenores_5 + nmenores_6_11 + 
                        nmenores_12_17 + nocupados + nincapacitados + ntrabajo_menores + 
                        Head_Mujer + Head_Afiliado_SS + P5140 + Npersug +
                        Head_exper_ult_trab + Head_Rec_alimento + Head_Rec_subsidio + 
                        Head_Rec_vivienda + Head_Ocupacion + maxEducLevel + Head_Primas +
                        Head_Segundo_trabajo + DormitorXpersona + Ln_Cuota + Head_Oficio +
                        Ln_Pago_arrien + nmujeres + Ocup_vivienda + 
                        Head_Cot_pension + Cabecera,
                      data = train, 
                      method = "xgbTree", 
                      trControl = fitControl,
                      tuneGrid=grid_xbgoost
)         

Xgboost_tree


test<- test  %>% mutate(Ingtotug_hat=predict(Xgboost_tree,newdata = test))

# Para medir el F1 primero creemos la variable de pobreza predicha

test <- test %>% mutate(Pobre_hat = ifelse(Ingtotug_hat<Lp,"Yes","No"))
test$Pobre_hat <- factor(test$Pobre_hat)

#Ahora sí podemos extraer el F1 del confusionmatrix
confusionMatrix(data = test$Pobre_hat, 
                reference = test$Pobre, positive="Yes", mode = "prec_recall")

#F1 = 0.35

#Representemos gráficamente los árboles entrenados
p_load(DiagrammeR)
tree_plot <- xgb.plot.tree(model = Xgboost_tree$finalModel,
                           trees = 1:2, plot_width = 1000, plot_height = 500)
tree_plot

# Ahora hagamos la predicción en la data de test hogares

predictSample <- test_hogares1   %>% 
  mutate(Ingtotug_hat = predict(Xgboost_tree, newdata = test_hogares1))

predictSample <- predictSample %>% mutate(Pobre = ifelse(Ingtotug_hat<Lp,"Yes","No"))
predictSample$Pobre <- factor(predictSample$Pobre)

predictSample <- predictSample %>% select(id,Pobre)

predictSample<- predictSample %>% 
  mutate(pobre=ifelse(Pobre=="Yes",1,0)) %>% 
  select(id,pobre)

#Kaggle puntaje = 
write.csv(predictSample,"income_prediction_xgboosting_ale1.csv", row.names = FALSE)

#- 4 | Modelo 2: XGBoost predicción del ingreso con Ln_ingpcug --------------------------

### Arreglo de data ###

#Seleccionamos las variables que necesitamos para predecir el ingreso
train_hogares1 <- dplyr::select(train_hogares, Ln_Ing_tot_hogar_per_cap, Pobre, Dominio, Depto, P5010, 
                                N_cuartos_hog, Nper, nmenores_5, nmenores_6_11, 
                                nmenores_12_17, nocupados, nincapacitados, ntrabajo_menores, 
                                Head_Mujer, Head_Afiliado_SS, P5140, Npersug, Lp, Li,
                                Head_exper_ult_trab, Head_Rec_alimento, Head_Rec_subsidio, 
                                Head_Rec_vivienda, Head_Ocupacion, maxEducLevel, Head_Primas,
                                Head_Segundo_trabajo, DormitorXpersona, Ln_Cuota, Head_Oficio,
                                Ln_Pago_arrien, nmujeres, Ocup_vivienda, 
                                Head_Cot_pension,Cabecera)

train_hogares1 <- train_hogares1 %>% 
  dplyr::mutate(
    Head_Mujer = factor(Head_Mujer),
    Head_Afiliado_SS = factor(Head_Afiliado_SS),
    Head_exper_ult_trab = factor(Head_exper_ult_trab),
    Head_Rec_alimento = factor(Head_Rec_alimento),
    Head_Rec_subsidio = factor(Head_Rec_subsidio),
    Head_Rec_vivienda = factor(Head_Rec_vivienda),
    Head_Ocupacion = factor(Head_Ocupacion),
    Head_Primas = factor(Head_Primas),
    Head_Segundo_trabajo = factor(Head_Segundo_trabajo),
    Head_Oficio = factor(Head_Oficio),
    Ocup_vivienda = factor(Ocup_vivienda),
    Head_Cot_pension = factor(Head_Cot_pension),
    Head_Segundo_trabajo = factor(Head_Segundo_trabajo),
    Cabecera = factor(Cabecera))


#Seleccionamos las mismas variables en la data de test
test_hogares1 <- test_hogares %>% 
  dplyr::select(id,Dominio, Depto, P5010, 
                N_cuartos_hog, Nper, nmenores_5, nmenores_6_11, 
                nmenores_12_17, nocupados, nincapacitados, ntrabajo_menores, 
                Head_Mujer, Head_Afiliado_SS, P5140, Npersug, Lp, Li,
                Head_exper_ult_trab, Head_Rec_alimento, Head_Rec_subsidio, 
                Head_Rec_vivienda, Head_Ocupacion, maxEducLevel, Head_Primas,
                Head_Segundo_trabajo, DormitorXpersona, Ln_Cuota, Head_Oficio,
                Ln_Pago_arrien, nmujeres, Ocup_vivienda, 
                Head_Cot_pension,Cabecera) %>% 
  mutate(
    Head_Mujer = factor(Head_Mujer),
    Head_Afiliado_SS = factor(Head_Afiliado_SS),
    Head_exper_ult_trab = factor(Head_exper_ult_trab),
    Head_Rec_alimento = factor(Head_Rec_alimento),
    Head_Rec_subsidio = factor(Head_Rec_subsidio),
    Head_Rec_vivienda = factor(Head_Rec_vivienda),
    Head_Ocupacion = factor(Head_Ocupacion),
    Head_Primas = factor(Head_Primas),
    Head_Segundo_trabajo = factor(Head_Segundo_trabajo),
    Head_Oficio = factor(Head_Oficio),
    Ocup_vivienda = factor(Ocup_vivienda),
    Head_Cot_pension = factor(Head_Cot_pension),
    Head_Segundo_trabajo = factor(Head_Segundo_trabajo),
    Cabecera = factor(Cabecera))

## Quitar infinitos ##

#Eliminemos los infinitos en la variable de ln del ingreso

train_hogares1 <- train_hogares1 %>% mutate(Ln_Ing_tot_hogar_per_cap = ifelse(Ln_Ing_tot_hogar_per_cap == "-Inf",0,Ln_Ing_tot_hogar_per_cap)) 

#revisemos que la variable no tenga infinitos

prueba <- train_hogares1 %>% group_by(Ln_Ing_tot_hogar_per_cap) %>% summarise(n())
rm(prueba)

#Convirtamos las variables categóricas en numéricas

sapply(train_hogares1,class)

train_hogares1 <- train_hogares1 %>% mutate(Dominio = as.numeric(Dominio),
                                            Depto = as.numeric(Depto),
                                            Head_Mujer = as.numeric(Head_Mujer),
                                            Head_Afiliado_SS = as.numeric(Head_Afiliado_SS),
                                            Head_exper_ult_trab = as.numeric(Head_exper_ult_trab),
                                            Head_Rec_alimento = as.numeric(Head_Rec_alimento),
                                            Head_Rec_subsidio = as.numeric(Head_Rec_subsidio),
                                            Head_Rec_vivienda = as.numeric(Head_Rec_vivienda),
                                            Head_Ocupacion = as.numeric(Head_Ocupacion),
                                            maxEducLevel = as.numeric(maxEducLevel),
                                            Head_Primas = as.numeric(Head_Primas),
                                            Head_Segundo_trabajo = as.numeric(Head_Segundo_trabajo),
                                            Head_Oficio = as.numeric(Head_Oficio),
                                            Ocup_vivienda = as.numeric(Ocup_vivienda),
                                            Head_Cot_pension = as.numeric(Head_Cot_pension),
                                            Cabecera = as.numeric(Cabecera))

test_hogares1 <- test_hogares1 %>% mutate(Dominio = as.numeric(Dominio),
                                          Depto = as.numeric(Depto),
                                          Head_Mujer = as.numeric(Head_Mujer),
                                          Head_Afiliado_SS = as.numeric(Head_Afiliado_SS),
                                          Head_exper_ult_trab = as.numeric(Head_exper_ult_trab),
                                          Head_Rec_alimento = as.numeric(Head_Rec_alimento),
                                          Head_Rec_subsidio = as.numeric(Head_Rec_subsidio),
                                          Head_Rec_vivienda = as.numeric(Head_Rec_vivienda),
                                          Head_Ocupacion = as.numeric(Head_Ocupacion),
                                          maxEducLevel = as.numeric(maxEducLevel),
                                          Head_Primas = as.numeric(Head_Primas),
                                          Head_Segundo_trabajo = as.numeric(Head_Segundo_trabajo),
                                          Head_Oficio = as.numeric(Head_Oficio),
                                          Ocup_vivienda = as.numeric(Ocup_vivienda),
                                          Head_Cot_pension = as.numeric(Head_Cot_pension),
                                          Cabecera = as.numeric(Cabecera))

#Revisemos rápidamente
sapply(train_hogares1,class)
sapply(train_hogares1,class)


# Dividimos la muestra para entrenar al modelo
set.seed(91519) # Importante definir la semilla. 

inTrain <- createDataPartition(
  y = train_hogares1$Ln_Ing_tot_hogar_per_cap,## La variable dependiente u objetivo 
  p = .7, ## Usamos 70%  de los datos en el conjunto de entrenamiento 
  list = FALSE)


train <- train_hogares1[ inTrain,]
test  <- train_hogares1[-inTrain,]

#Ajuste del modelo

fitControl<-trainControl(method ="cv",
                         number=5)

#Cargamos los parámetros del boosting
grid_xbgoost <- expand.grid(nrounds = c(250),
                            max_depth = c(4),
                            eta = c(0.01), 
                            gamma = c(0), 
                            min_child_weight = c(10, 25),
                            colsample_bytree = c(0.33,0.66), 
                            subsample = c(0.4))
grid_xbgoost


#Entrenamos el modelo
set.seed(91519) # Importante definir la semilla antes entrenar
Xgboost_tree <- train(Ln_Ing_tot_hogar_per_cap~Dominio + Depto + P5010 + P5010^2 + 
                        N_cuartos_hog + Nper + nmenores_5 + nmenores_6_11 + 
                        nmenores_12_17 + nocupados + nincapacitados + ntrabajo_menores + 
                        Head_Mujer + Head_Afiliado_SS + Npersug + Npersug^2 +
                        Head_Rec_subsidio + Head_Rec_vivienda + maxEducLevel + 
                        Head_Segundo_trabajo + DormitorXpersona^2 + Ln_Cuota + Head_Oficio +
                        Ln_Pago_arrien + nmujeres^2 + Ocup_vivienda + 
                        Head_Cot_pension + Cabecera,
                      data = train, 
                      method = "xgbTree", 
                      trControl = fitControl,
                      tuneGrid=grid_xbgoost
)         

Xgboost_tree


test<- test  %>% mutate(Ln_Ing_tot_hogar_per_cap_hat=predict(Xgboost_tree,newdata = test))

#Marquemos los que tienen ingreso cero

test <- test %>% mutate(Cero = ifelse(Ln_Ing_tot_hogar_per_cap==0,1,0))

#Pasemos a exponencial el ingreso

test <- test %>% mutate(Ing_tot_hogar_per_cap_hat=exp(Ln_Ing_tot_hogar_per_cap))

# Para medir el F1 primero creemos la variable de pobreza predicha

test <- test %>% mutate(Pobre_hat = ifelse(Ing_tot_hogar_per_cap_hat<Lp*Npersug,"Yes","No"))
test$Pobre_hat <- factor(test$Pobre_hat)

#Ahora sí podemos extraer el F1 del confusionmatrix
confusionMatrix(data = test$Pobre_hat, 
                reference = test$Pobre, positive="Yes", mode = "prec_recall")

#F1 = 0.40

#Representemos gráficamente los árboles entrenados
tree_plot <- xgb.plot.tree(model = Xgboost_tree$finalModel,
                           trees = 1:2, plot_width = 1000, plot_height = 500)
tree_plot

## Ahora hagamos la predicción en la data de test hogares

predictSample <- test_hogares1   %>% 
  mutate(Ln_Ing_tot_hogar_per_cap_hat = predict(Xgboost_tree, newdata = test_hogares1))

#Pasemos a exponencial el ingreso

predictSample <- predictSample %>% mutate(Ing_tot_hogar_per_cap_hat=exp(Ln_Ing_tot_hogar_per_cap_hat))

predictSample <- predictSample %>% mutate(Pobre = ifelse(Ing_tot_hogar_per_cap_hat<Lp*Npersug,"Yes","No"))
predictSample$Pobre <- factor(predictSample$Pobre)

predictSample <- predictSample %>% dplyr::select(id,Pobre)

predictSample<- predictSample %>% 
  mutate(pobre=ifelse(Pobre=="Yes",1,0)) %>% 
  dplyr::select(id,pobre)

#Kaggle puntaje = 
write.csv(predictSample,"income_prediction_ln_ingpcug_xgboosting_ale3.csv", row.names = FALSE)

