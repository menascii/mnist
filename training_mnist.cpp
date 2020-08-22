/* mnist training program in c++ by @menascii
     
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
*/

#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

void read_headers(ifstream &image, ifstream &label);
void read_image(ifstream &image, ifstream &label, int image_digit[28][28], double *layer_one, double expected[]);
void get_digit(ifstream &image, int image_digit[28][28], double *layer_one);
void print_digit(int image_digit[28][28]);
void get_label(ifstream &label, int image_digit[28][28], double expected[]);
double* init_layer(int layer_size);
double** set_weights(int input_layer_size, int output_layer_size);
double ** init_deltas(int input_layer_size, int output_layer_size);
double sigmoid(double x);
void perceptron(double *layer_one, double *layer_two, double *layer_three, 
                double **layer_one_weights, double **layer_two_weights);
double cost_value(double *layer_three, double expected[]);
void back_propagation(double *layer_one, double *layer_two, double *layer_three, 
                      double **layer_one_weights, double **layer_two_weights,
                      double **layer_one_deltas, double **layer_two_deltas, 
                      double *layer_two_thetas, double *layer_three_thetas, double expected[]);
void training_process(double *layer_one, double *layer_two, double *layer_three, 
                     double **layer_one_weights, double **layer_two_weights,
                     double **layer_one_deltas, double **layer_two_deltas,
                     double *layer_two_thetas, double *layer_three_thetas,
                     double expected[]);
void write_weights(string file_name, double **layer_one_weights, double **layer_two_weights);

int main()
{
  ifstream training_image;
  ifstream training_label;
     
  // mnist binary training digits file name                                       
  string training_images = "train-images";
  // mnist binary training labels                                                                                    
  string training_labels = "train-labels";

  // output weights file name                                                           
  string model_weights = "model-weights";
  // mnist image digit 28 x 28                              
  int image_digit[28][28];
  double expected[10];
  // neural network layers
  double *layer_one, *layer_two, *layer_three;
  // neural network weights
  double **layer_one_weights, **layer_two_weights;
  // neural network derivatives
  double **layer_one_deltas, **layer_two_deltas;
  // neural network biases
  double *layer_two_thetas, *layer_three_thetas;
    
  // open mnist training images
  training_image.open(training_images.c_str(), ios::in | ios::binary);
  // open mnist training labels
  training_label.open(training_labels.c_str(), ios::in | ios::binary);
  // read headers from training images and labels
  read_headers(training_image, training_label);

  // initialize neural network layers
  layer_one = init_layer(784);
  layer_two = init_layer(128);
  layer_three = init_layer(10);

  // seed random number
  srand(time(0));
  // initialize and set neural network weights
  layer_one_weights = set_weights(784, 128);
  layer_two_weights = set_weights(784, 128);

  // initialize neural network derivatives
  layer_one_deltas = init_deltas(784, 128);
  layer_two_deltas = init_deltas(128, 10);

  // initialize neural network biases
  layer_two_thetas = new double [128];
  layer_three_thetas = new double [10];

  // training loop to iterate each mnist digit and label
  for (int image_index = 0; image_index < 60000; image_index++)
    {
      cout << "mnist training image #: " << image_index << endl; 
      // read image and label
      // assign each 28*28 pixel to first layer as 784 activations in neural network
      read_image(training_image, training_label, image_digit, layer_one, expected);
      training_process(layer_one, layer_two, layer_three, 
                       layer_one_weights, layer_two_weights, 
                       layer_one_deltas, layer_two_deltas,
                       layer_two_thetas, layer_three_thetas,
                       expected);
    	
      // update weights every 100 iterations to save cpu cycles
    	if (image_index % 100 == 0)
    	  {
    	 	    write_weights(model_weights, layer_one_weights, layer_two_weights);
    	  }
  }
  
  training_image.close();
  training_label.close();
  return 0;
}

void read_headers(ifstream &training_image, ifstream &training_label)
{
  // strip headers from training files
  char number;
  for (int i = 0; i < 16; i++)
    {
      training_image.read(&number, sizeof(char));
    }
  for (int i = 0; i < 8; i++)
    {
      training_label.read(&number, sizeof(char));
    }
}

double* init_layer(int layer_size)
{
  double *layer = new double[layer_size];
  return layer;
}

double **set_weights(int input_layer_size, int output_layer_size)
{
  // set weights values from input layer to output layer
  double **layer_weights = new double *[input_layer_size];  
  for (int i = 0; i < input_layer_size; i++)
    {
      layer_weights[i] = new double [output_layer_size];
      for (int j = 0; j < output_layer_size; j++)
        {
          int sign = rand() % 2;
          layer_weights[i][j] = (double)(rand() % 6) / 10.0;
          if (sign == 1)
            {
              layer_weights[i][j] = - layer_weights[i][j];
            }
        }
    }
  return layer_weights;
}

double **init_deltas(int input_layer_size, int output_layer_size)
{
  double **layer_deltas = new double *[input_layer_size];
  for (int i = 0; i < input_layer_size; i++)
    {
      layer_deltas[i] = new double [output_layer_size];
      for (int j = 0; j < output_layer_size; j++)
        {
          layer_deltas[i][j] = 0.0;
        }
    }
  return layer_deltas;
}

void read_image(ifstream &image, ifstream &label, int image_digit[28][28], double *layer_one, double expected[])
{
  // read 28x28 binary mnist image
  get_digit(image, image_digit, layer_one);
  // read binary mnist label 
  get_label(label, image_digit, expected);
  // print mnist 28x28 mnist digit and label
  print_digit(image_digit);
}

void get_digit(ifstream &image, int image_digit[28][28], double *layer_one)
{
  // read 28x28 image one character at a time
  char number;
  for (int j = 0; j < 28; j++)
    {
      for (int i = 0; i < 28; i++)
        {
          int layer_index = i + j * 28;
          image.read(&number, sizeof(char));
          if (number == 0)
            {
                image_digit[i][j] = 0;
            }
          else
            {
                image_digit[i][j] = 1;
            }
            layer_one[layer_index] = image_digit[i][j];
        }
    }
}

void print_digit(int image_digit[28][28])
{
  cout << "####### training digit #######" << endl;
  for (int j = 0; j < 28; j++)
  {
    for (int i = 0; i < 28; i++)
      {
        cout << image_digit[i][j];
      }
    cout << endl;
  }
  cout << endl;
}

void get_label(ifstream &label, int image_digit[28][28], double expected[])
{
  // read training label value
  char number;
  label.read(&number, sizeof(char));
  for (int i = 0; i < 10; i++)
    {
      expected[i] = 0.0;
    }
  expected[number] = 1.0;
  cout << "mnist label: " << (int)(number) << endl;
}

void training_process(double *layer_one, double *layer_two, double *layer_three, 
                     double **layer_one_weights, double **layer_two_weights, 
                     double **layer_one_deltas, double **layer_two_deltas,
                     double *layer_two_thetas, double *layer_three_thetas,
                     double expected[])
{
  // difference between expected value and predicted value
  double cost_error = 0;
  // minimize cost error using threshold value
  double epsilon = .0027;
  // iterate for gradient descent to minmize cost error
  for (int i = 0; i < 512; i++)
  {
    perceptron(layer_one, layer_two, layer_three, 
               layer_one_weights, layer_two_weights);

    back_propagation(layer_one, layer_two, layer_three, 
                     layer_one_weights, layer_two_weights, 
                     layer_one_deltas, layer_two_deltas,
                     layer_two_thetas, layer_three_thetas, expected);

    // return cost error value
    cost_error = cost_value(layer_three, expected); 
    // minimize cost error value
    if (cost_error  < epsilon)
      {
        break;
      }
  }
}

void perceptron(double *layer_one, double *layer_two, double *layer_three, 
                double **layer_one_weights, double **layer_two_weights)
{  
  // neural network z values
  double *layer_one_zs, *layer_two_zs;  
  // initialize z values for neural network
  layer_one_zs = new double [128];
  for (int i = 0; i < 128; i++)
    {
      layer_one_zs[i] = 0.0;
        // multiply and sum all weight and pixel values
      for (int j = 0; j < 784; j++)
    	{
    	  layer_one_zs[i] += layer_one[j] * layer_one_weights[j][i];
    	}
      // sigmoid z return value
      layer_two[i] = sigmoid(layer_one_zs[i]);
    }

  layer_two_zs = new double [10];  
  for (int i = 0; i < 10; i++)
    {
      layer_two_zs[i] = 0.0;
      for (int j = 0; j < 128; j++)
        {
          layer_two_zs[i] += layer_two[j] * layer_two_weights[j][i];
        }
      layer_three[i] = sigmoid(layer_two_zs[i]);
    }
}

void back_propagation(double *layer_one, double *layer_two, double *layer_three, 
                      double **layer_one_weights, double **layer_two_weights,
                      double **layer_one_deltas, double **layer_two_deltas,
                      double *layer_two_thetas, double *layer_three_thetas,
                      double expected[])
{ 
  double bias;
  double momentum = 0.66;
  double learning_rate = .027;

  // thetas are used as biases to multiply with inputs from the previous layer
  for (int i = 0; i < 10; i++)
    {
      layer_three_thetas[i] = layer_three[i] * (1 - layer_three[i]) * (expected[i] - layer_three[i]);
    }

  for (int i = 0; i < 128; i++)
    {
      bias = 0.0;
      for (int j = 0; j < 10; j++)
        {
          bias += layer_two_weights[i][j] * layer_three_thetas[j];
          // deltas use the change between target and current values to train model
          layer_two_deltas[i][j] = (learning_rate * layer_three_thetas[j] * layer_two[i]) + (momentum * layer_two_deltas[i][j]);
          // update weights
          layer_two_weights[i][j] += layer_two_deltas[i][j];
        }
      layer_two_thetas[i] = layer_two[i] * (1 - layer_two[i]) * bias;
    }

  for (int i = 0; i < 784; i++)
    {
      for (int j = 0 ; j < 128 ; j++ )
        {
          layer_one_deltas[i][j] = (learning_rate * layer_two_thetas[j] * layer_one[i]) + (momentum * layer_one_deltas[i][j]);
          layer_one_weights[i][j] += layer_one_deltas[i][j];
        }
    }
}

double sigmoid(double x)
{
    double sigmoid_value = 0.0;
    sigmoid_value = (1.0 / (1.0 + exp(-x)));
    return sigmoid_value;
}

double cost_value(double *layer_three, double expected[])
{
  double cost = 0.0;
  for (int i = 0; i < 10; i++)
    {
      cost += (layer_three[i] - expected[i]) * (layer_three[i] - expected[i]);
    }
  return cost;
}

void write_weights(string file_name, double **layer_one_weights, double **layer_two_weights)
{
  ofstream file;
  file.open(file_name.c_str(), ios::out);
  cout << "upating weights to " << file_name << " file" << endl;
  // first input layer to hidden middle layer                                                                                                     
  for (int i = 0; i < 784; i++)
    {
      for (int j = 0; j < 128; j++)
    	{
    	   file << layer_one_weights[i][j] << " ";
    	}
      file << endl;
  }
  // hidden middle layer to third output layer                                                                                                                              
  for (int i = 0; i < 128; i++)
    {
      for (int j = 0; j < 10; j++)
    	{
    	  file << layer_two_weights[i][j] << " ";
    	}
      file << endl;
    } 
   file.close();
}
