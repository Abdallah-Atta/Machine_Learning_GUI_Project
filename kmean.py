from sklearn.cluster import KMeans
from kneed import KneeLocator
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from tkinter import messagebox


def K_Mean_function(df):
    sse_list = []
    for k in range(1, 11):
        kmean = KMeans(n_clusters=k, init="random", max_iter=300, random_state=1, n_init=10)
        kmean.fit(df)  # Fit the model to the scaled data
        sse_list.append(kmean.inertia_)

    # Find the elbow point
    k1 = KneeLocator(range(1, 11), sse_list, curve='convex', direction='decreasing')
    optimal_num_clusters = k1.elbow

    # print("Optimal number of clusters:", optimal_num_clusters)


    best_kmean = KMeans(n_clusters=optimal_num_clusters, init='random', random_state=1,max_iter=300,n_init=10)
    best_kmean.fit(df) 



    #accurcy 
    cluster_labels = best_kmean.labels_
    silhouette_avg = silhouette_score(df,cluster_labels)
    
    messagebox.showinfo("Info",
                        "Number_of_clusters : "  + str(optimal_num_clusters) + "\n" +
                        "Accuracy : " + str(silhouette_avg) + "\n" )

    # prediction
    # cluster_labels = best_kmean.fit_predict(df)
    # preds = best_kmean.labels_
    # kmean_df=pd.DataFrame(df)  
    # kmean_df['kmean_cluster'] = preds

    #save new dataset
    # kmean_df.to_csv('kmean_result.csv',index=False)
    # print("The new dataset is saved")



