from tkinter import *
from tkinter.ttk import *

tk=Tk()
popup = Toplevel()
Label(popup, text="Files being downloaded").grid(row=0,column=0)
progress=Progressbar(popup,orient=HORIZONTAL,length=100,mode='determinate')

def bar():
    import time
    progress['value']=20
    tk.update_idletasks()
    time.sleep(1)
    progress['value']=50
    tk.update_idletasks()
    time.sleep(1)
    progress['value']=80
    tk.update_idletasks()
    time.sleep(1)
    progress['value']=100

progress.grid(row = 1)
Button(tk,text='foo',command=bar).pack()
mainloop()