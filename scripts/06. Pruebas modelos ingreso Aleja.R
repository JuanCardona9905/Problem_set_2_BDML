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

#- 1 | Pruebas varias ------------------------------------------------

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
#- 2 | Modelo 1: XGBoost con variables relevantes ---------------------

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



# Dividimos la muestra para entrenar al modelo
set.seed(91519) # Importante definir la semilla. 

inTrain <- createDataPartition(
  y = credit$Default,## La variable dependiente u objetivo 
  p = .7, ## Usamos 70%  de los datos en el conjunto de entrenamiento 
  list = FALSE)


train <- credit[ inTrain,]
test  <- credit[-inTrain,]

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
Xgboost_tree <- train(Default~duration+amount+installment+age+
                        history+purpose+foreign+rent,
                      data = train, 
                      method = "xgbTree", 
                      trControl = ctrl,
                      tuneGrid=grid_xbgoost,
                      metric = "ROC",
                      verbosity = 0
)         

Xgboost_tree

#Obtenemos el AUC para este modelo
pred_prob <- predict(Xgboost_tree,
                     newdata = test, 
                     type = "prob")   


aucval_XGboost <- Metrics::auc(actual = default,predicted = pred_prob[,2])
aucval_XGboost 


#Representemos gráficamente os árboles entrenados
p_load(DiagrammeR)
tree_plot <- xgb.plot.tree(model = Xgboost_tree$finalModel,
                           trees = 1:2, plot_width = 1000, plot_height = 500)
tree_plot





