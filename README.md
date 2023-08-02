# Face-Recognition-Using-CNN
I have successfully completed a small face recognition project using a Convolutional Neural Network (CNN) algorithm. The CNN demonstrated a high accuracy rate, making it suitable for face recognition. To create the face data, I recorded videos and converted them into images. These images were then processed to extract the faces, which were stored in separate folders on Google Drive, labeled with the respective individuals' names.

Due to limited GPU access, I had to optimize the data to reduce memory size. To do this, I combined all the data into 'x' and 'y', which were subsequently divided into training and testing sets for further evaluation.

Upon feeding the data into the CNN algorithm, the model achieved an impressive accuracy of approximately 0.9154. I validated the results using a confusion matrix, which turned out to be perfect, as expected.

The model was saved and downloaded for future use. To test its functionality, I implemented it with OpenCV2, and the model was able to successfully detect faces and provide the corresponding names of the individuals.
