# Ex.No: 13 Sign Language Detection by using the CNN  
### DATE: 24/10/2024                                                                            
### REGISTER NUMBER : 212222220029
### AIM: 
To build a Convolutional Neural Network (CNN) model for sign language detection, train the model on a dataset of hand gesture images, and evaluate its performance in predicting sign language symbols.

###  Algorithm:
#### 1. Data Preparation:
Collect and preprocess the dataset of sign language hand gestures.
Normalize the images by scaling pixel values to the range [0, 1].
Split the dataset into training and testing sets.

#### 2.Model Design:
Design a CNN model with multiple convolutional, pooling, and fully connected layers to extract features from hand gesture images.
Use a softmax layer for the output to classify the gestures into different sign language classes.

#### 3.Model Compilation:
Compile the model with a suitable optimizer (such as RMSprop or Adam), categorical cross-entropy loss (for multi-class classification), and accuracy as the evaluation metric.
Training the Model:

#### 4.Train the model on the training dataset.
Use early stopping to prevent overfitting and save the best model weights.

#### 5.Model Evaluation:
Evaluate the model on the test dataset to check its accuracy and generalization capability.

#### 6.Prediction:
Use the trained model to predict and classify new hand gestures.


### Program:

```
import numpy as np # linear algebra
import pandas as pd

# %%
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %%
train_df = pd.read_csv('C:/Users/SEC/Desktop/sign language/sign_mnist_train.csv')
test_df = pd.read_csv('C:/Users/SEC/Desktop/sign language/sign_mnist_test.csv')


# %%
train_df.info()


# %%
test_df.info()


# %%
train_df.describe()

# %%
train_df.head(6)

# %%
train_label=train_df['label']
train_label.head()
trainset=train_df.drop(['label'],axis=1)
trainset.head()

# %%
X_train = trainset.values
X_train = trainset.values.reshape(-1,28,28,1)
print(X_train.shape)

# %%
test_label=test_df['label']
X_test=test_df.drop(['label'],axis=1)
print(X_test.shape)
X_test.head()

# %%
from sklearn.preprocessing import LabelBinarizer
lb=LabelBinarizer()
y_train=lb.fit_transform(train_label)
y_test=lb.fit_transform(test_label)

# %%
y_train

# %%
X_test=X_test.values.reshape(-1,28,28,1)

# %%
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

# %%
train_datagen = ImageDataGenerator(rescale = 1./255,
                                  rotation_range = 0,
                                  height_shift_range=0.2,
                                  width_shift_range=0.2,
                                  shear_range=0,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  fill_mode='nearest')

X_test=X_test/255

# %%
fig,axe=plt.subplots(2,2)
fig.suptitle('Preview of dataset')
axe[0,0].imshow(X_train[0].reshape(28,28),cmap='gray')
axe[0,0].set_title('label: 3  letter: C')
axe[0,1].imshow(X_train[1].reshape(28,28),cmap='gray')
axe[0,1].set_title('label: 6  letter: F')
axe[1,0].imshow(X_train[2].reshape(28,28),cmap='gray')
axe[1,0].set_title('label: 2  letter: B')
axe[1,1].imshow(X_train[4].reshape(28,28),cmap='gray')
axe[1,1].set_title('label: 13  letter: M')

# %%
sns.countplot(train_label)
plt.title("Frequency of each label")

# %%
model=Sequential()
model.add(Conv2D(128,kernel_size=(5,5),
                 strides=1,padding='same',activation='relu',input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(3,3),strides=2,padding='same'))
model.add(Conv2D(64,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
model.add(MaxPool2D((2,2),2,padding='same'))
model.add(Conv2D(32,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
model.add(MaxPool2D((2,2),2,padding='same'))
          
model.add(Flatten())

# %%
model.add(Dense(units=512,activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(units=24,activation='softmax'))
model.summary()

# %%
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# %%
model.fit(train_datagen.flow(X_train,y_train,batch_size=200),
         epochs = 35,
          validation_data=(X_test,y_test),
          shuffle=1
         )

# %%
(ls,acc)=model.evaluate(x=X_test,y=y_test)

# %%
print('MODEL ACCURACY = {}%'.format(acc*100))

# %%
import numpy as np
import cv2
from keras.models import load_model
import pyttsx3

# Load the trained model
model = load_model(r'C:\Users\SEC\Desktop\sign language\cnn8grps_rad1_model.h5')

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (400, 400))  # Resize to 400x400
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Keep 3 channels for RGB
    img = img / 255.0  # Normalize
    img = img.reshape(-1, 400, 400, 3)  # Reshape to (1, 400, 400, 3)
    return img

# Function to predict the letter from the image
def predict_sign_language(image_path):
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions)
    return predicted_class

# Function to convert letter index to character
def index_to_letter(index):
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return letters[index]

# Function to convert text to speech
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Example usage
image_path = r'C:\Users\SEC\Desktop\sign language\WhatsApp Image 2024-11-10 at 22.38.37_5938f3ca.jpg'
predicted_letter_index = predict_sign_language(image_path)
predicted_letter = index_to_letter(predicted_letter_index)

# Convert the predicted letter to text
text_output = predicted_letter

# Print the predicted text
print(f'The predicted sign language letter is: {text_output}')

# Convert the text to speech
speak_text(f'The predicted letter is {text_output}')

```


### Output:

![Screenshot 2024-11-11 160210](https://github.com/user-attachments/assets/9874d784-3870-4bc3-97c2-78bfc24e7f0d)

![Screenshot 2024-11-11 160149](https://github.com/user-attachments/assets/73c65043-4e82-4b17-b690-1d7d342b5346)

![Screenshot 2024-11-11 160125](https://github.com/user-attachments/assets/756ed36e-ad2c-4dbf-9cb1-b5566c2e8a48)



### Result:
Thus the system was trained successfully and the prediction was carried out.
