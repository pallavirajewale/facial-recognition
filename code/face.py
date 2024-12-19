import face_recognition
import tkinter as tk
from tkinter import filedialog

# Create a simple GUI window
root = tk.Tk()
root.withdraw()  # Hide the main window

# Prompt the user to select an image file
file_path = filedialog.askopenfilename(title="Select an Image File", 
                                        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])

# Check if a file was selected
if file_path:  # If the user selected a file
    try:
        # Load the image file
        image_of_person = face_recognition.load_image_file(file_path)

        # Proceed with your processing
        print("Image loaded successfully!")
    except Exception as e:
        print(f"Error loading image: {e}")
else:
    print("No file selected.")
