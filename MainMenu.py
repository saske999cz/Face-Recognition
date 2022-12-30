import tkinter as tk
from tkinter import *
from webbrowser import *
from PIL import ImageTk, Image
from subprocess import call
import os




def minimize():
    window.iconify()

def logout():
    window.destroy()
    # os.startfile("E:/Project/PBL4/Login.py")
    RunApp("E:/Project/PBL4/Login.py")
     
def RunApp(path):
    call(["python", path])

def addnew():
    window.withdraw()
    RunApp("E:/Project/PBL4/app.py")   
    window.deiconify()  
def run():
    window.withdraw()
    RunApp("E:/Project/PBL4/main.py")
    window.deiconify()
    
def runServer():
    window.withdraw()
    RunApp("E:/Project/PBL4/server.py")       
    window.deiconify()

def train():
    window.withdraw()
    RunApp("E:/Project/PBL4/dataProcessing.py")
    RunApp("E:/Project/PBL4/trainModel.py")  
    window.deiconify()  
    
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
add_img = Image.open("E:/Project/PBL4/App-ui/Img/add.png")
face_img = Image.open("E:/Project/PBL4/App-ui/Img/run.png")
logout_img = Image.open("E:/Project/PBL4/App-ui/Img/logout.png")
server_img = Image.open("E:/Project/PBL4/App-ui/Img/server.png")
train_img = Image.open("E:/Project/PBL4/App-ui/Img/train.png")
minimize_img = Image.open("E:/Project/PBL4/App-ui/Img/minimize.png")

resized = ImageTk.PhotoImage(bg.resize((800,500), Image.Resampling.LANCZOS))
resized1 = add_img.resize((220,150), Image.Resampling.LANCZOS)
resized2 = face_img.resize((220,150), Image.Resampling.LANCZOS)
resized3 = logout_img.resize((220,150), Image.Resampling.LANCZOS)
resized4 = server_img.resize((220,150), Image.Resampling.LANCZOS)
resized5 = train_img.resize((220,150), Image.Resampling.LANCZOS)
resized6 = minimize_img.resize((64,53), Image.Resampling.LANCZOS)

add_btn = ImageTk.PhotoImage(resized1)
face_btn = ImageTk.PhotoImage(resized2)
logout_btn =ImageTk.PhotoImage(resized3)
server_btn = ImageTk.PhotoImage(resized4)
train_btn = ImageTk.PhotoImage(resized5)
minimize_btn = ImageTk.PhotoImage(resized6)


img_add = Label(image=add_btn)
img_face = Label(image=face_btn)
img_logout = Label(image=logout_btn)
img_server = Label(image=server_btn)
img_train = Label(image=train_btn)
img_minimize = Label(image=minimize_btn)


canvas1 = Canvas( window, width = 400,
				height = 800)

canvas1.pack(fill = "both", expand = True)

# Display image
canvas1.create_image( 0, 0, image = resized,
					anchor = "nw")



# Create Buttons
button1 = Button( window, image = add_btn, text = "Add", command=addnew, borderwidth = 0)
button2 = Button( window, image = face_btn, text = "Face", command=run, borderwidth = 0)
button3 = Button( window, image = server_btn, text = "Server", command=runServer, borderwidth = 0)
button4 = Button(window, image = logout_btn, text = "Logout", command=logout, borderwidth = 0)
button5 = Button(window, image = train_btn, text = "Train", command=train, borderwidth = 0)
button6 = Button( window, image = minimize_btn, text = "Minimize", command=minimize, borderwidth = 0)

button1.config(height=40, width=160)
button2.config(height=40, width=160)
button3.config(height=40, width=160)
button4.config(height=40, width=160)
button5.config(height=40, width=160)
button6.config(height=30, width=40)
# Display Buttons
button1_canvas = canvas1.create_window( 125, 50,
									anchor = "nw",
									window = button1)


button2_canvas = canvas1.create_window( 125, 210,
									anchor = "nw",
									window = button2)

button3_canvas = canvas1.create_window( 125, 290,
									anchor = "nw",
									window = button3)

button4_canvas = canvas1.create_window( 125, 370,
									anchor = "nw",
									window = button4)
button5_canvas = canvas1.create_window( 125, 130,
									anchor = "nw",
									window = button5)
button6_canvas = canvas1.create_window( 740, 10,
									anchor = "nw",
									window = button6)

#window.overrideredirect(True)
window.resizable(0,0)
window.iconbitmap(r'E:/Project/PBL4/face2.ico')
window.mainloop()