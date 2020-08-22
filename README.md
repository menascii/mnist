# mnist_database_digit_recognition
mnist database digit recognition

mnist training program in c++
     
   compile:
   
      g++ training_mnist.cpp
    
   run:
   
      ./a.out  

   input:     
   
      hardcoded file name for 60,000 mnist training images
      hardcoded file name for 60,000 mnist training labels
      download training data from http://yann.lecun.com/exdb/mnist/
        train-images-idx3-ubyte.gz
        train-labels-idx1-ubyte.gz
        
        unzip and rename accordingly for hardcoded filename values
   
   output:
   
      write updated model weights to file
      
   note:
   
      while this program is updating the weights you can execute testing_mnist.cpp
      to check the accuracy using the mnist test data. you can also just wait for the 
      process to complete and check the final output. 

      refer to testing_mnist.cpp for further details.

