##prueba modelos

rm(list = ls())
require("pacman")
p_load("tidyverse",
       "glmnet",
       "caret",
       "smotefamily",
       "dplyr",
       "dummy")

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
                metric = "F1",
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
  mutate(Pobre = predict(model1, newdata = test_hogares, type = "raw")  
  )  %>% select(id,Pobre)

predictSample<- predictSample %>% 
  mutate(pobre=ifelse(Pobre=="Yes",1,0)) %>% 
  select(id,pobre)

write.csv(predictSample,"classification_elasticnet_smote.csv", row.names = FALSE)
