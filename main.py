import tkinter as tk
from tkinter import messagebox
import sklearn
import pickle
import numpy as np
from sklearn import linear_model
from PIL import Image, ImageTk
from tkinter import Tk, Label


pickle_in = open("modelmain.pickle", "rb")
le = pickle.load(pickle_in)


def predict_grade():
    try:
        # Get input values
        grade1 = float(entry_grade1.get())
        grade2 = float(entry_grade2.get())
        study_time = int(entry_study_time.get())
        free_time = int(entry_free_time.get())
        absences = int(entry_absences.get())

        
        test = np.array([grade1, grade2, study_time, absences, free_time]).reshape(1, -1)
        pred = le.predict(test)[0]

        if pred>20:
            pred=20
    
        predicted_grade.set(f"Predicted Grade 3: {pred:.2f}")

        #Suggestions
        if pred <= 5.0:
            suggestion.set("Studies need serious attention.\nConsider increasing study time and investing more free time into studies.")
        elif 5.0 < pred < 10.0:
            suggestion.set("You need to work harder!\nConsider increasing study time and investing more free time into studies.")
        elif 10.0 <= pred < 15.0:
            suggestion.set("Decent performance. Consider improving somewhat more for better results.")
        elif 15.0 <= pred < 18.0:
            suggestion.set("Great work!\nConsider increasing study time and investing more free time into studies.\nStart taking out more time for hobbies!")
        else:
            suggestion.set("Excellent!! You are doing great. Keep it up.\nConsider increasing study time and investing more free time into studies.\nStart taking out more time for hobbies!")
        if absences > 12:
            suggestion.set(suggestion.get() + "\nNeed to improve attendance.")

    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values.")


root = tk.Tk()
root.title("Student Grade Predictor")
root.geometry("600x600")
root.resizable(False, False)
root.configure(bg="#f0f0f0")


img = Image.open("tk2.jpeg")  
img = img.resize((100, 100))
img = ImageTk.PhotoImage(img)


image_label = tk.Label(root, image=img, bg="#f0f0f0")
image_label.pack(pady=10)

# Header
header_label = tk.Label(root, text="Student Grade Predictor", font=("Helvetica", 16), bg="#f0f0f0")
header_label.pack(pady=10)

# Input Fields
input_frame = tk.Frame(root, bg="#f0f0f0")
input_frame.pack(pady=10)

label_grade1 = tk.Label(input_frame, text="Grade 1 (Out of 20):", bg="#f0f0f0")
label_grade1.grid(row=0, column=0, padx=5)
entry_grade1 = tk.Entry(input_frame, width=10)
entry_grade1.grid(row=0, column=1, padx=5)

label_grade2 = tk.Label(input_frame, text="Grade 2 (Out of 20):", bg="#f0f0f0")
label_grade2.grid(row=1, column=0, padx=5)
entry_grade2 = tk.Entry(input_frame, width=10)
entry_grade2.grid(row=1, column=1, padx=5)

label_study_time = tk.Label(input_frame, text="Study Time (hours):", bg="#f0f0f0")
label_study_time.grid(row=2, column=0, padx=5)
entry_study_time = tk.Entry(input_frame, width=10)
entry_study_time.grid(row=2, column=1, padx=5)

label_free_time = tk.Label(input_frame, text="Free Time (hours):", bg="#f0f0f0")
label_free_time.grid(row=3, column=0, padx=5)
entry_free_time = tk.Entry(input_frame, width=10)
entry_free_time.grid(row=3, column=1, padx=5)

label_absences = tk.Label(input_frame, text="Absences (In number of days):", bg="#f0f0f0")
label_absences.grid(row=4, column=0, padx=5)
entry_absences = tk.Entry(input_frame, width=10)
entry_absences.grid(row=4, column=1, padx=5)

# Predict Button
predict_button = tk.Button(root, text="Predict", command=predict_grade, font=("Helvetica", 14), bg="#4CAF50", fg="white")
predict_button.pack(pady=10)

# Output
predicted_grade = tk.StringVar()
predicted_grade_label = tk.Label(root, textvariable=predicted_grade, font=("Helvetica", 14), bg="#f0f0f0")
predicted_grade_label.pack()

suggestion = tk.StringVar()
suggestion_label = tk.Label(root, textvariable=suggestion, font=("Helvetica", 12), bg="#f0f0f0")
suggestion_label.pack()

root.mainloop()
