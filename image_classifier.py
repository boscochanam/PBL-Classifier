import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import tensorflow as tf
import numpy as np
import os
import shutil
from datetime import datetime

# Loading Trained Model
model = tf.keras.models.load_model('model1.h5')

# Define the paths to the output directories for the sorted images
output_dir_1 = 'Signature'
output_dir_2 = 'Portrait'
output_dir_unsure = 'Unsure'

# Create the output directories if they don't already exist
if not os.path.exists(output_dir_1):
    os.makedirs(output_dir_1)
if not os.path.exists(output_dir_2):
    os.makedirs(output_dir_2)
if not os.path.exists(output_dir_unsure):
    os.makedirs(output_dir_unsure)

class ImageClassifier:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Image Classifier')
        self.root.configure(bg='#F0F8FF')  # Set background color
        self.root.geometry('500x500')

        self.images = []

        # Create a button to select images
        select_button = tk.Button(self.root, text='Select Images', command=self.select_images, bg='#A9A9A9')
        select_button.pack(pady=10)

        self.root.mainloop()

    def select_images(self):
        # Open a folder dialog to allow the user to select a folder containing image files
        folder_path = filedialog.askdirectory()

        # Load the selected images and display them in a new window
        self.new_win = tk.Toplevel(self.root)
        self.new_win.title('Selected Images')

        canvas = tk.Canvas(self.new_win, bg='#F0F8FF')
        canvas.pack(side='left', fill='both', expand=True)

        scrollbar = tk.Scrollbar(self.new_win, orient='vertical', command=canvas.yview)
        scrollbar.pack(side='right', fill='y')

        frame = tk.Frame(canvas, bg='#F0F8FF')
        canvas.create_window((0,0), window=frame, anchor='nw')
        frame.bind('<Configure>', lambda event, canvas=canvas: canvas.configure(scrollregion=canvas.bbox('all')))

        file_paths = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'):
                file_path = os.path.join(folder_path, file_name)
                img = Image.open(file_path)
                resize = img.resize((256, 256))
                photo = ImageTk.PhotoImage(resize)
                self.images.append(photo)
                file_paths.append(file_path)

        for i, photo in enumerate(self.images):
            label = tk.Label(frame, image=photo)
            label.grid(row=i//3, column=i%3, padx=10, pady=10)

        # Resize the canvas to fit the images
        frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox('all'))

        self.file_paths = file_paths

        # Add a button for classification
        classify_button = tk.Button(self.new_win, text='Classify', command=self.classify_images, bg='#FFC0CB')
        classify_button.pack(pady=10)

    from datetime import datetime  # Import datetime class from datetime module

    def classify_images(self):

        # Loop over each selected image and preprocess it for input to the model
        for file_path in self.file_paths:
            img = cv2.imread(file_path)
            resize = tf.image.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (256,256))

            # Replace with the path to your model
            yhat = model.predict(np.expand_dims(resize/255, 0))

            # Copy the image to the appropriate output directory based on its predicted class
            output_dir_1 = 'Signature' # Replace with the path to your output directory
            output_dir_2 = 'Portrait' # Replace with the path to your output directory
            output_dir_unsure = 'unsure' # Replace with the path to your output directory

            if yhat > 0.5:
                output_path = os.path.join(output_dir_1, os.path.basename(file_path))
                label = "sig"  # Set label as "sig"
            elif yhat < 0.5:
                output_path = os.path.join(output_dir_2, os.path.basename(file_path))
                label = "por"  # Set label as "por"
            else:
                output_path = os.path.join(output_dir_unsure, os.path.basename(file_path))
                label = "unsure"  # Set label as "unsure"

            # Get current date, hour, minute, second, and month
            now = datetime.now()
            date_time = now.strftime("%f_%S_%H_%d_%b")[:-3]

            # Update the output filename with uploadername, label, and date_time
            uploadername = os.path.basename(file_path).split(" - ")[-1].split(".")[0]
            output_filename = f"{uploadername}({label}){date_time}.jpg"
            output_path = os.path.join(os.path.dirname(output_path), output_filename)

            shutil.copyfile(file_path, output_path)


            
        # Close the confirmation window
        self.root.destroy()

app = ImageClassifier()


