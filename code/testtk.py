import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()  # Hide the root window to only show the file dialog
dataset_folder = filedialog.askdirectory(title="Select Dataset Folder")
print("Selected folder:", dataset_folder)
