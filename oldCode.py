import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# Specify the path to your dataset directory
data_dir = "/Users/aleksandrlukasuk/aitu_study/computer_vision/ass_1/Agricultural-crops"

# Create an ImageDataGenerator for loading and preprocessing the images
datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalize pixel values to the range [0, 1]
    validation_split=0.1  # 10% of the data will be used for validation
)

# Load and preprocess the dataset using the generator
batch_size = 32  # Adjust as needed
image_size = (224, 224)  # Adjust based on your image size
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse',  # Use 'sparse' for integer labels
    subset='training',  # Specify 'training' to get the training set
    seed=seed
)

# Split data into train, validation, and test sets
X_train_filenames, X_test_filenames, y_train, y_test = train_test_split(
    train_generator.filenames, train_generator.classes,
    test_size=0.1, random_state=seed
)
X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(
    X_train_filenames, y_train,
    test_size=0.1, random_state=seed
)

# Define Softmax Regression model
def softmax_regression(input_dim, num_classes, alpha=0.1, num_epochs=1000):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=alpha),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val))
    
    return model

# Define SVM model
def support_vector_machine(X_train, y_train, X_val, y_val, C=1.0):
    model = SVC(C=C, kernel='linear')
    model.fit(X_train, y_train)
    
    return model

# Load and preprocess image data for X_train, X_val, and X_test
X_train = np.array([img_to_array(load_img(os.path.join(data_dir, filename), target_size=image_size)) for filename in X_train_filenames])
X_val = np.array([img_to_array(load_img(os.path.join(data_dir, filename), target_size=image_size)) for filename in X_val_filenames])
X_test = np.array([img_to_array(load_img(os.path.join(data_dir, filename), target_size=image_size)) for filename in X_test_filenames])

# Flatten the image data
X_train = X_train.reshape(X_train.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Define Softmax Regression model as a scikit-learn estimator
class SoftmaxRegressionEstimator(BaseEstimator):
    def __init__(self, input_dim, num_classes, alpha=0.1, num_epochs=1000):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.alpha = alpha
        self.num_epochs = num_epochs

    def fit(self, X, y):
        self.model = softmax_regression(self.input_dim, self.num_classes, self.alpha, self.num_epochs)
        return self.model

    def predict(self, X):
        # return self.model.predict_classes(X
        return np.argmax(self.model.predict(X), axis=-1)

# Define SVM model as a scikit-learn estimator
class SVMEstimator(BaseEstimator):
    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        self.model = support_vector_machine(X_train, y_train, X_val, y_val, self.C)
        return self.model

    def predict(self, X):
        return self.model.predict(X)

# Perform 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
softmax_scores = cross_val_score(SoftmaxRegressionEstimator(X_train.shape[1], len(np.unique(y_train)), alpha=0.1, num_epochs=1000), X_train, y_train, cv=cv, scoring='accuracy')
svm_scores = cross_val_score(SVMEstimator(C=1.0), X_train, y_train, cv=cv, scoring='accuracy')

# Print cross-validation results
print("Cross-validation results (Softmax Regression):", softmax_scores)
print("Cross-validation results (SVM):", svm_scores)

# Train the best model on the entire training set (you can use the model with the highest cross-validation accuracy)
best_softmax_model = SoftmaxRegressionEstimator(X_train.shape[1], len(np.unique(y_train)), alpha=0.1, num_epochs=1000)
best_softmax_model.fit(X_train, y_train)
best_svm_model = SVMEstimator(C=1.0)
best_svm_model.fit(X_train, y_train)

# Evaluate on the test set
softmax_test_accuracy = accuracy_score(y_test, best_softmax_model.predict(X_test))
svm_test_accuracy = accuracy_score(y_test, best_svm_model.predict(X_test))

# Print test set results
print("Test set accuracy (Softmax Regression):", softmax_test_accuracy)
print("Test set accuracy (SVM):", svm_test_accuracy)

# Choose the algorithm with the best accuracy and explain it
if np.mean(softmax_scores) > np.mean(svm_scores):
    best_model = best_softmax_model
    best_model_name = "Softmax Regression"
else:
    best_model = best_svm_model
    best_model_name = "SVM"

print("The best algorithm is:", best_model_name)
