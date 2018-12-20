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


import csv

url = 'https://raw.githubusercontent.com/dquarshie89/Data-620/master/movie_plots.csv'
response = urlopen(url)
cr = csv.reader(response)

for row in cr:
    print(row)
