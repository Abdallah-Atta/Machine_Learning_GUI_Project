from tkinter import * 
from tkinter import filedialog, messagebox
import pandas as pd 
from tkinter import ttk
from suprvised_Pre_prossesing import *
from unsuprvised_Pre_prossesing import *
from linear_regression import * 
from decision_tree import *
from SVM import *
from kmean import *

myframe1= Tk()

myframe1.title("GUI")
myframe1.geometry("500x300")

def supervised_browse_file():
    global filename
    filename = filedialog.askopenfilename()
     

    if 'filename' in globals() and filename:
        global label
        label=Label(frame_to_get_file,text="select the target colum ")
        label.place(x=70,y=140)
        items=return_columns_names()
        selected_item = StringVar()
        global dropdown
        dropdown = ttk.Combobox(frame_to_get_file, textvariable=selected_item)
        dropdown['values'] = items
        dropdown.place(x=70,y=160)


def read_file(filename):
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filename)
            return df
       
        elif filename.endswith('.xls') or filename.endswith('.xlsx'):
            df = pd.read_excel(filename)
            return df
    except:
        messagebox.showerror("Erorr",'File Extension is Not Supported')
        return 0


def return_columns_names():
    try:
        #if 'filename' in globals() and filename:
            df=read_file(filename)
            return list(df.columns.sort_values()) 
    except:
         messagebox.showerror("Erorr",'File Extension is Not Supported')


def get_colum_from_box():
    x=str(dropdown.get()) 
    return x

    



def run_suprvised_preprossing_funchon():
    global newdata
    newdata=suprvised_preprocessing(read_file(filename), get_colum_from_box())
    return newdata

def running_linear_Regression():
   Linear_Regression_function(newdata,get_colum_from_box())
    
def running_linear_SVM():
    support_vector_machine_funchon(newdata,get_colum_from_box(),'linear')
    
def running_deciton_tree():
    decision_tree_funchon(newdata,get_colum_from_box())

def running_Nonlinear_SVM():
    support_vector_machine_funchon(newdata,get_colum_from_box(),'rbf')

def open_tap_for_file_for_subervised_learnig():
    global frame_to_get_file
    frame_to_get_file=Tk()
    frame_to_get_file.title("file upload ")
    frame_to_get_file.geometry("300x300")

    mylapel=Label(frame_to_get_file,text="upload the file ")
    mylapel.pack()

    #buttom for browse 
    buttom_for_browse_file=Button(frame_to_get_file,text=" browse ",width=15,height=2,pady=10,command=supervised_browse_file)
    buttom_for_browse_file.pack()
    
    #buttom for next
    buttom_for_next=Button(frame_to_get_file,text=" Next >",width=10,height=1,command= open_frame_for_models_subervised_models )
    buttom_for_next.place(x=200,y=250)

    buttom_for_exit=Button(frame_to_get_file,text="exit",width=10,height=1,command=exit)
    buttom_for_exit.place(x=10,y=250)
        
    
    
  


    frame_to_get_file.mainloop()
    # buttom_for_preprossing=Button(frame_to_get_file,text="preprossing",width=10,height=1,command=run_suprvised_preprossing_funchon)
    # buttom_for_preprossing.place(x=105,y=250)
 
    # targrtColum=Entry()



 
 
def open_frame_for_models_subervised_models():
    
    if 'filename' in globals() and filename:
        frame_for_subervised_learning=Tk()
        frame_for_subervised_learning.title("Subervised Learning Models ")
        frame_for_subervised_learning.geometry("500x500")
        
        run_suprvised_preprossing_funchon()

        #buttom for Linear Regression
        butoom_for_LinearRegression=Button(frame_for_subervised_learning,text="Linear Regression" ,width=35,height=4,command=running_linear_Regression)
        butoom_for_LinearRegression.grid(row=1,column=0,padx=100,pady=50)
    
      
        butoom_for_linear_SVM=Button(frame_for_subervised_learning,text=" Linear SVM ",width=35,height=4 , command= running_linear_SVM)
        butoom_for_linear_SVM.grid(row=2,column=0)


        butoom_for_DecisionTrees=Button(frame_for_subervised_learning,text=" Decision tree ",width=35,height=4,command=running_deciton_tree)
        butoom_for_DecisionTrees.grid(row=3,column=0,padx=100,pady=50)


        butoom_for_NonLinear_SVM=Button(frame_for_subervised_learning,text=" Non Linear SVM ",width=35,height=4 ,command= running_Nonlinear_SVM)
        butoom_for_NonLinear_SVM.grid(row=4,column=0)


        


    else:
        print()
        messagebox.showwarning("showwarning","you should select the file ")

# =================================================================================================================================


def unsupervised_browse_file():
    global filename
    filename = filedialog.askopenfilename()
     
def run_unsuprvised_preprossing_funchon():
    global newdata
    newdata=unsupervised_preprocessing(read_file(filename))
    return newdata


def running_Kmean():
   K_Mean_function(newdata)
    
#open file upload tap for unsubervised learning
def open_tap_for_file_for_unsubervised_learning():
    frame_to_get_file=Tk()
    frame_to_get_file.title("file upload ")
    frame_to_get_file.geometry("300x300")

    mylapel=Label(frame_to_get_file,text="upload the file ")
    mylapel.pack()

    #buttom for browse 
    buttom_for_browse_file=Button(frame_to_get_file,text=" browse ",width=15,height=2,pady=10,command=unsupervised_browse_file)
    buttom_for_browse_file.pack()

    #buttom for next
    buttom_for_next_forunsubervisedFrame=Button(frame_to_get_file,text=" Next >",width=10,height=1,command=open_frame_for_models_Unsubervised_models)
    buttom_for_next_forunsubervisedFrame.place(x=200,y=250)

    buttom_for_exit=Button(frame_to_get_file,text="exit",width=10,height=1,command=exit)
    buttom_for_exit.place(x=10,y=250)
    



# this frame if user clicked on unsubervised learning buttom 
def open_frame_for_models_Unsubervised_models():
    
    if 'filename' in globals() and filename:
        frame_for_Unsubervised_learning=Tk()
        frame_for_Unsubervised_learning.title("UnSubervised Learning Models ")
        frame_for_Unsubervised_learning.geometry("500x500")
    
        run_unsuprvised_preprossing_funchon()
   
        #buttom for clustring 
        butoom_for_Kmean=Button(frame_for_Unsubervised_learning,text="Kmean" ,width=35,height=4 , command=running_Kmean)
        butoom_for_Kmean.grid(row=1,column=0,padx=100,pady=50)

        # #buttom for Decision Trees
        # butoom_for_DecisionTrees=Button(frame_for_Unsubervised_learning,text="Decision tree ",width=35,height=4  )
        # butoom_for_DecisionTrees.grid(row=2,column=0)

    else:
        print()
        messagebox.showwarning("showwarning","you should select the file ")





#buttom for subervised_learning this open new tap for subervised_learning models 
butoom_for_subervised_learning=Button(myframe1,text=" Supervised learning ",width=35,height=4,command=open_tap_for_file_for_subervised_learnig)
butoom_for_subervised_learning.grid(row=1,column=0,pady=50,padx=100)


#buttom for Unsubervised_learning this open new tap for Unsubervised_learning models 
butoom_for_unsubervised_learning=Button(myframe1,text="UnSupervised learning ",width=35,height=4,command=open_tap_for_file_for_unsubervised_learning)
butoom_for_unsubervised_learning.grid(row=10,column=0)



myframe1.mainloop()







