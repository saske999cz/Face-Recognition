import tkinter as tk
from tkinter import *
from webbrowser import *
from PIL import ImageTk, Image
from subprocess import call
from tkinter import messagebox
import os
import pyrebase


firebaseConfig ={
    'apiKey': "AIzaSyCg5xApDAFPFp942uNhq341kKvZ_qgA21g",
    'authDomain': "facerecognitionrealtime.firebaseapp.com",
    'databaseURL': "https://facerecognitionrealtime-default-rtdb.firebaseio.com",
    'projectId': "facerecognitionrealtime",
    'storageBucket': "facerecognitionrealtime.appspot.com",
    'messagingSenderId': "736773951815",
    'appId': "1:736773951815:web:9d245328d585424e373c62",
    'measurementId': "G-444RV84W04"
}

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

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



def RunApp():
    call(["python", "E:/Project/PBL4/MainMenu.py"])

def minimize():
    window.iconify()

def login():
    if username_input.get() == "" or password_input.get() == "":
        messagebox.showerror("Invalid input", "Vui lòng điền thông tin đăng nhập")
    else:
        try:
            login = auth.sign_in_with_email_and_password(username_input.get(), password_input.get())
            check = True
        except:
            messagebox.showerror("Invalid input", "Sai mật khẩu hoặc tài khoản")
            check = False
        if(check == True):
            window.destroy()
            RunApp()
       
    
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


bg = ImageTk.PhotoImage(Image.open("E:/Project/PBL4/App-ui/Img/background.jpg"))
login_img = Image.open("E:/Project/PBL4/App-ui/Img/log.png")
exit_img = Image.open("E:/Project/PBL4/App-ui/Img/exit.png")
minimize_img = Image.open("E:/Project/PBL4/App-ui/Img/minimize.png")



resized1 = login_img.resize((220,220), Image.Resampling.LANCZOS)
resized2 = exit_img.resize((64,53), Image.Resampling.LANCZOS)
resized3 = minimize_img.resize((64,53), Image.Resampling.LANCZOS)



login_btn = ImageTk.PhotoImage(resized1)
exit_btn = ImageTk.PhotoImage(resized2)
minimize_btn = ImageTk.PhotoImage(resized3)


img_login = Label(image=login_btn)
img_exit = Label(image=exit_btn)
img_minimize = Label(image=minimize_btn)

canvas1 = Canvas( window, width = 400,
				height = 800)

canvas1.pack(fill = "both", expand = True)

# Display image
canvas1.create_image( 0, 0, image = bg,
					anchor = "nw")



# Create Buttons
button1 = Button( window, image = exit_btn, text = "Exit", command=window.destroy, borderwidth = 0)
button2 = Button( window, image = login_btn, text = "Login", command=login, borderwidth = 0)
button3 = Button( window, image = minimize_btn, text = "Minimize", command=minimize, borderwidth = 0)

button1.config(height=30, width=40)
button2.config(height=40, width=160)
button3.config(height=30, width=40)

# Display Buttons
button1_canvas = canvas1.create_window( 740, 455,
									anchor = "nw",
									window = button1)


button2_canvas = canvas1.create_window( 575, 300,
									anchor = "nw",
									window = button2)

button3_canvas = canvas1.create_window( 10, 10,
									anchor = "nw",
									window = button3)


username_input = EntryWithPlaceholder(window, "Email")
username_input.config(width=18, font='Space_Mono 18')
password_input = EntryWithPlaceholder(window, "Password")
password_input.config(width=18, font='Space_Mono 18')
password_input.config(show="*")


username_input.place(x =535, y = 180)
password_input.place(x =535, y = 240)

window.resizable(0,0)
window.iconbitmap(r'E:/Project/PBL4/face2.ico')
window.mainloop()