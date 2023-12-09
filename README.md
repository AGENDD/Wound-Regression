# Wound-Regression

This is a group project for CISC-3023. The project requirement is to predict wound information in the images using different models. The authors' information is shown below:

JIANG Rui	  DC027649

MAI Jiajun	DC127853

This project contains three folders, corresponding to each model. Some important files are explained below:

\NeuralNetwork

   train.py: training the model
   
   test.py: testing the model
   
   Augmentation.py: augmenting the data

   neuralNetwork_aug.pth: The model file training on the augmented data
   
   neuralNetwork.pth: the model file training on the origional data
   
  (please run the Augmentation.py before the train.py if the "\Training_Aug" folder is empty)

\RandomForests

  bRT_test.py: training the model(bRT)
  
  bRT_train.py: testing the model(bRT)
  
  RF_test.py: training the model(RT)
  
  RF_train.py: testing the model(RT)

  RT.sav: the saved RandomForests model.

  boostRT.sav: the saved boost RandomForests model.

\SVR

   train.py: training the model
   
   test.py: testing the model

   SVR.sav: the saved model.

All the test results visualization are stored in the "\Test_out" folder.
 

 
