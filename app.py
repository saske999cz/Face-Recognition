from tkinter import *
from subprocess import call
import tkinter as tk
from getData import getData
from webbrowser import *
from PIL import ImageTk, Image
from tkinter import messagebox
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from datetime import datetime
cred = credentials.Certificate("E:/Project/PBL4/serviceAccountKey.json")
firebase_admin.initialize_app(cred,
                              {'databaseURL': "https://facerecognitionrealtime-default-rtdb.firebaseio.com/"})

class EntryWithPlaceholder(tk.Entry):
    def __init__(self, master=None, placeholder="PLACEHOLDER", color='grey'):
        super().__init__(master)

        self.placeholder = placeholder
        self.placeholder_color = color
        self.default_fg_color = self['fg']

        self.bind("<FocusIn>", self.foc_in)
        self.bind("<FocusOut>", self.foc_out)

        self.put_placeholder()

    def put_placeholder(self):
        self.insert(0, self.placeholder)
        self['fg'] = self.placeholder_color

    def foc_in(self, *args):
        if self['fg'] == self.placeholder_color:
            self.delete('0', 'end')
            self['fg'] = self.default_fg_color

    def foc_out(self, *args):
        if not self.get():
            self.put_placeholder()


def clicked():
    if name_input.get() == "" or name_input.get() == "Enter Name":
        messagebox.showwarning("Invalid input", "Vui lòng nhập tên")
    else:
        window.withdraw()
        ref = db.reference('Users')
        data = {
            name_input.get():
            {
            "name": name_input.get(),
            "last_checked": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
                }
        
        for key, value in data.items():
            ref.child(key).set(value)
        getData(name_input.get())
        call(["python","E:/Project/PBL4/dataProcessing.py"])
        call(["python","E:/Project/PBL4/trainModel.py"])
        window.deiconify()
        
        
        
    
def back():
    window.destroy()
    
    
def SaveLastClickPos(event):
    global lastClickX, lastClickY
    lastClickX = event.x
    lastClickY = event.y


def Dragging(event):
    x, y = event.x - lastClickX + window.winfo_x(), event.y - lastClickY + window.winfo_y()
    window.geometry("+%s+%s" % (x , y))

    
lastClickX = 0
lastClickY = 0    
window = tk.Tk()
apps = []

window.title("Face Recognition App")
window.geometry("800x500+300+100")
window.attributes('-topmost', True)
window.bind('<Button-1>', SaveLastClickPos)
window.bind('<B1-Motion>', Dragging)


bg = Image.open("E:/Project/PBL4/App-ui/Img/mainimg.png")
add_img = Image.open("E:/Project/PBL4/App-ui/Img/data.png")
back_img = Image.open("E:/Project/PBL4/App-ui/Img/back.png")

resized = ImageTk.PhotoImage(bg.resize((800,500), Image.Resampling.LANCZOS))
resized1 = add_img.resize((42,42), Image.Resampling.LANCZOS)
resized2 = back_img.resize((60,60), Image.Resampling.LANCZOS)

add_btn = ImageTk.PhotoImage(resized1)
back_btn = ImageTk.PhotoImage(resized2)



img_add = Label(image=add_btn)
img_back = Label(image=back_btn)



canvas1 = Canvas( window, width = 400,
				height = 800)

canvas1.pack(fill = "both", expand = True)

# Display image
canvas1.create_image( 0, 0, image = resized,
					anchor = "nw")



# Create Buttons
button1 = Button( window, image = add_btn, text = "Add", command=clicked, borderwidth = 0)
button2 = Button( window, image = back_btn, text = "Back", command=back, borderwidth = 0)


button1.config(height=26, width=30)
button2.config(height=40, width=40)

# Display Buttons
button1_canvas = canvas1.create_window( 530, 100,
									anchor = "nw",
									window = button1)


button2_canvas = canvas1.create_window( 20, 15,
									anchor = "nw",
									window = button2)




name_input = EntryWithPlaceholder(window, "Enter Name")
name_input.config(width=20, font="Arial 16")

name_input.place(x =280, y = 100)





#window.overrideredirect(True)
window.resizable(0,0)
window.iconbitmap(r'E:/Project/PBL4/face2.ico')
window.mainloop()