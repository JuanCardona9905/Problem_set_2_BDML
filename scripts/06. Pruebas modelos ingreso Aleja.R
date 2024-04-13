########### Modelos de prueba de predicción indirecta ##############
########### con métodos CARTs, Random Forest y Boosting ################

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
       "xgboost",
       "ROSE",#remuestreo ROSE
       "ranger") #random forest 
library("dplyr")

setwd("C:/Users/LENOVO/Documents/Aleja Maestría/Big data y machine learning/Problem_set_2_BDML/Data")
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

#- 3 | Modelo 1: XGBoost con variables relevantes ---------------------

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

#- 4 | Modelo 3: XGBoost predicción del ingreso con Ln_ingpcug --------------------------

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
Xgboost_tree <- train(Ln_Ing_tot_hogar_per_cap~Dominio + Depto + P5010 + P5010^2 + 
                        N_cuartos_hog + Nper + nmenores_5 + nmenores_6_11 + 
                        nmenores_12_17 + nocupados + nincapacitados + ntrabajo_menores + 
                        Head_Mujer + Head_Afiliado_SS + Npersug + Npersug^2
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
