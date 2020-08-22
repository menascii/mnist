# mnist_database_digit_recognition
mnist database digit recognition

mnist training program in c++
     
   compile:
      g++ training_mnist.cpp
    
   run:
      ./a.out  

   input:      
      hardcoded file name for mnist training images containing 60,000 digits
      hardcoded file name for corresponding 60,000 mnist training labels
      download training data from http://yann.lecun.com/exdb/mnist/
        train-images-idx3-ubyte.gz
        train-labels-idx1-ubyte.gz
        
        unzip and rename accordingly for hardcoded filestream values
   
   output:
      write updated model weights to file used to test mnist test data using testing_mnist.cpp
      
   note:
      while this program is updating the weights you can execute testing_mnist.cpp
      to check the training progress. you can also just wait for the process to complete 
      and check the final output. refer to testing_mnist.cpp for further details.
