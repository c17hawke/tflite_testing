# First, load you model and allocate the tensors
import tensorflow.lite as tflite
# Load TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path='model.tflite')
#allocate the tensors
interpreter.allocate_tensors()

# Get the input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Read in a sample image you want the model to predict and decode to a tensor
import cv2
import numpy as np
import imghdr
image_path='sunflower.jpeg'
if imghdr.what(image_path) not in ["jpeg", "jpg", "png"]:
    print("quiting the program")
img = cv2.imread(image_path)
img = cv2.resize(img,(224,224))
#Preprocess the image to required size and cast
input_shape = input_details[0]['shape']
input_tensor= np.array(np.expand_dims(img,0))
# Set the tensor to point to the input data to be inferred, then run the inference
input_index = interpreter.get_input_details()[0]["index"]
interpreter.set_tensor(input_index, input_tensor)
interpreter.invoke()
output_details = interpreter.get_output_details()
# Make the prediction
output_data = interpreter.get_tensor(output_details[0]['index'])
pred = np.squeeze(output_data)

# If you print “pred”, you will get an array of uint8 (0–255) values. These correspond to the confidence for each class. In my example that would be the 16 bird species we specified in the Collecting Image Data For Machine Learning in Python tutorial.
# To get a more human friendly prediction, specify the class indices below.
class_ind = {
  0: "daisy",
  1: "dandelion",
  2: "roses",
  3: "sunflowers",
  4: "tulips",}
# # Since “pred” corresponds to the confidence of each bird class, our prediction should be the highest value in the array. Grab the highest prediction location.
highest_pred_loc = np.argmax(pred)
print(highest_pred_loc, pred)
# # Use the “highest_pred_loc” to search the “class_ind” dictionary for the actual bird’s name. Print out the prediction and you’re done!
flower_name = class_ind[highest_pred_loc]
print(f"predicted flower is {flower_name}")