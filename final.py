#Import all the required packages
import pandas as pd
import numpy as np
from IPython.display import display # Allows the use of display() for DataFrames
import time
import pickle #To save the objects that were created using webscraping
import pprint
from IPython.display import HTML
from lxml import html
import requests
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from urllib.request import urlopen
from bs4 import BeautifulSoup

import os

import os
import re
import nltk
import string
from collections import Counter


from nltk.corpus import stopwords
from nltk import PorterStemmer as Stemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

url = pd.read_csv('https://raw.githubusercontent.com/dquarshie89/Data-620/master/movie_plots.csv')


df = url.sample(5000)
df.shape

def cleaning(text):
    text = text.lower() #Make all the text lowercase
    text = ''.join([t for t in text if t not in string.punctuation]) #Get rid of puntucations
    text = [t for t in text.split() if t not in stopwords.words('english')] #Remove stop words
    st = Stemmer() #Stem sentences to reduce inflections(https://en.wikipedia.org/wiki/Stemming)
    text = [st.stem(t) for t in text]
    return text

v = TfidfVectorizer(decode_error='replace', encoding='utf-8')

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = v.fit_transform(df['Plot'].values.astype('U'))
print(tfidf_matrix.shape)


cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

#Convert cos_sim (a numpy array) to a data frame with rows and columns as movie IDs
cos_sim_df = pd.DataFrame(cos_sim,columns=df["Movie ID"].tolist(),index=df["Movie ID"].tolist())

display(cos_sim_df)

#Get the mapping between available Movie plots and movie IDs
Movie_Map= df
#display(Movie_Map.head())
def Get_Recommendations(Movie_ID,cos_sim_df):
    recommended_idx=np.argpartition(np.array(cos_sim_df[Movie_ID].tolist()), -6)[-6:]
    #print(np.array(cos_sim_df[Movie_ID].tolist())[recommended_idx])
    Recommended_Movie_IDs = cos_sim_df.columns[recommended_idx].tolist()
    #return Recommended_Movie_IDs
    return dict(zip(Recommended_Movie_IDs,np.array(cos_sim_df[Movie_ID].tolist())[recommended_idx]))

def Get_Available_Images():
    
    image_files = os.listdir("./images")
    #Make sure that we are dealing with movie data files only
    image_files = [i for i in image_files if re.search('[1-9]*\.jpg',i)]
    y = list()
    for i in image_files:
        y.append(int(i.split(".")[0]))
    return y

def Display_Recommendations(Recommended_Movies_Dict,Movie_Map,Source_Movie_ID):
    #The following statement will make sure that we sort the movies in the descending order of similarity
    Recommended_Movies = pd.DataFrame(sorted(Recommended_Movies_Dict.items(), key=lambda x: -x[1]))[0].tolist()
    
    #Delete the liked movie from the list
    Recommended_Movies = Recommended_Movies[1:]
    
    Recommended_Movies_Plot = dict()
    for i in Recommended_Movies:
        Recommended_Movies_Plot[i] = Movie_Map[Movie_Map["Movie ID"] == i]["Plot"].tolist()[0]
    
    #Recommended_Movies=list(Recommended_Movies_Dict.keys())
    #Movie_Map[Movie_Map["Movie_ID"].isin(Recommended_Movies)]["Movie_ID"].tolist()
    Available_Images_List = Get_Available_Images()
    Source_Movie_Name = Movie_Map[Movie_Map["Movie ID"] == Source_Movie_ID]["Title"].tolist()[0]
    Source_Plot = Movie_Map[Movie_Map["Movie ID"] == Source_Movie_ID]["Plot"].tolist()[0]
    print("Assuming that the user liked {}:".format(Source_Movie_Name))
    
    #Recommended_Movies = list(set(Recommended_Movies) - set([Source_Movie_ID]))
    
    if Source_Movie_ID in Available_Images_List:
        #print("The user has liked {}".format(Source_Movie_Name))
        display(HTML("<table><tr><td><img src='./images/"+str(Source_Movie_ID)+".jpg' title='"+str(Source_Plot)+"'></td></tr></table>" \
            ))        
        
    display_html = ""
    display_values = ""
    for i in Recommended_Movies:
        if i in Available_Images_List:
            display_html = display_html + "<td><img src='./images/"+str(i)+".jpg' title='"+str(Recommended_Movies_Plot[i])+"'></td>"
            display_values = display_values + "<td> Similarity:"+str(Recommended_Movies_Dict[i])+"</td>"
    print("The following movies are recommended:")        
    display(HTML("<table><tr>"+display_html+"</tr><tr>"+display_values+"</tr></table>" \
            ))        
    #return display_html            
    #Get available images for movies:
    
Recommended_Movies = Get_Recommendations(3974,cos_sim_df)
Recommended_Movies
Display_Recommendations(Recommended_Movies,Movie_Map,3974)
