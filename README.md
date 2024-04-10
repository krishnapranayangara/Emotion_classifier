# Emotion_classifier
The project discussed is based on the Image classification problem using deep convolution neural networks.


Through web scraping using beautiful soup framework, happy and sad data is collected. The Source would be taking in the request URL and based on the source link would collect images in that webpage. We could modify this URL to include webpage 2,3 and 4 to collect data. The images would then be labeled based on the image download number and stored in the image path specified by us. A total of 480 images have been considered regardless of the image resolution.

The data has been split into train test and validation data points. The data sets thus will be stored in the image batches (ibatch), to be fed to the model along with their labels. The Image classification is binary classifier that takes in the Boolean values of 0 and 1. 1 being sad and 0 being happy. Model 1 has comparatively better results where the loss has been gradually decreasing to 0. The image batches are then fed into the model with the specified hyper parameters. The summary of operations would be stored in the logs using TensorFlow call backs. I have run the model for 45 epochs with a default batch size of 32. Learning rate would be 0.01. Interpolation for the image dataset would be bilinear interpolation. Interpolation helps in scaling the data for better learning and prediction capabilities. Model1 steadily improves in accuracy for 45 epochs and reaches a stagnation. Model 2 performs well but fails to achieve accuracy greater than 60%. This is due to the vanishing gradient issue discussed in the analysis section. Model 3 has improved accuracy and performance with a precision of over 70% due to the tanh activation yet does not exceed model 1 accuracy scores.

![image](https://github.com/krishnapranayangara/Emotion_classifier/assets/33367492/66b4a538-db8f-44b2-b6e0-3edde7bb3ed6)


![image](https://github.com/krishnapranayangara/Emotion_classifier/assets/33367492/7fef6091-2b0c-486f-b8f1-72ddd65b09cd)


![image](https://github.com/krishnapranayangara/Emotion_classifier/assets/33367492/c3154630-2907-4b15-b9b8-999e55b8403b)


The deep learning neural networks are improved neural networks with more hidden layers. These perform exceptionally well with substantial amounts of data. However, the activation functions such as sigmoid still suffer from a phenomenon known as vanishing gradient problem.
The neural networks will take in the input and pass them to hidden layers and gives out an output, through this process the forward propagation uses varied weights and biases. The weights are multiplied with the inputs and the aggregate sum is added to bias which in turn goes through the activation function to give us the output. Backward propagation is the next step after forward propagation. A loss function gets calculated using the prediction from the forward propagation learnings as shown in figure 13. Here the y depicts the prediction and mx+b would be the output with m being weight and b the bias.


![image](https://github.com/krishnapranayangara/Emotion_classifier/assets/33367492/e1e06acf-709e-4e2c-aa19-ce48df82d419)

The weights and biases are updated accordingly for a reduced loss function. At each iteration, the loss function is calculated using the derivative of earlier input and learning rate. The formula for calculating the gradient descent can be seen in figure14.


![image](https://github.com/krishnapranayangara/Emotion_classifier/assets/33367492/bc8fb1b3-c0ee-4869-9d6c-5d20dcf99555)

All the new weights are dependent on the old weights, since the main criterion for gradient descent is finding the minima and reducing the loss function, the gradient of loss function based on the derivatives of earlier inputs would be steadily decreasing till it’s close to zero. This phenomenon is known as vanishing gradient problem. This problem has a significant impact for deep layers resulting in reduced accuracy. This problem persists for tanh and can be fixed only through the use Rectified linear unit function. Relu only has the derivative values of 0 and 1 thus giving us 0 only if the results are less than 0, this helps us prevent vanishing gradient problem.
However, from the figure (15) below we can see that the model 1 which uses relu quickly overfits the results to 1. This is due to the convergence, the very concept of relu is giving us a 0 or 1 output instead of some fraction between 0 and 1 like sigmoid and tanh. This means that the model will learn much faster and takes all the positive values and nullifies negative values. Thus, the model using relu converges much quicky than sigmoid and tanh giving us an accuracy of 100%.



![image](https://github.com/krishnapranayangara/Emotion_classifier/assets/33367492/70df2e88-6bcf-4b23-90b7-a69e36a73dc3)

To Summarize, we have extracted datasets from the internet using beautiful soup, a web scraping tool and used preprocessed the data through normalizing and resizing it to train our three different models of convolution neural networks. The data has been further split into training, testing and validation splits and were fed into the models. The model 1 performed much better than model 2 and 3. The reason for the reduced prediction accuracy of the latter models is due to the phenomenon known as vanishing gradient descent. We have generated an ROC curve to visualize the best model.


![image](https://github.com/krishnapranayangara/Emotion_classifier/assets/33367492/6323a9db-03cc-46cf-8b2f-622d55cd89c5)


From Figure 16, the best model is Relu model and thus we have used it for the prediction of a test image. The outputs have proven that the images have been predicted correctly.



![image](https://github.com/krishnapranayangara/Emotion_classifier/assets/33367492/c0e752c4-f7cf-4fc1-9ae1-f88e316fc338)


