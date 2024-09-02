import  pandas  as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , mean_squared_error ,recall_score,precision_score,confusion_matrix
from tkinter import messagebox

def decision_tree_funchon(df,targetCoulmn):
        

        x = df.drop(targetCoulmn, axis=1)  # Adjust 'target_column' to your target variable
        y = df[targetCoulmn]  # Adjust 'target_column' to your target variable

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

        # Instantiate and fit the model
        model = DecisionTreeClassifier()
        model.fit(x_train, y_train)
        y_pred=model.predict(x_test)
         
        accuracy = accuracy_score(y_test, y_pred)
        MSE = mean_squared_error(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        # recall = recall_score(y_test, y_pred)
        # precision =precision_score(y_test, y_pred)


        messagebox.showinfo("Info", 
                                "Accuracy: " + str(accuracy) + "\n" +
                                "MES: " + str(MSE) + "\n" +
                                "Confusion Matrix: " + "\n" + str(confusion) + "\n" )
                                # "Recall Score: " + str(recall) + "\n" +
                                # "Precision Score: " + str(precision))
