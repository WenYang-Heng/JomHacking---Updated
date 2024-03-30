import views
from flask import Flask, request, render_template, jsonify
import requests
import math
import PyPDF2
import numpy as np
import tabula
import re
import pandas as pd
import json
import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PyPDF2 import PdfReader
from tabula.io import read_pdf
from LogisticSGDModel import LogisticSGDModel
from NeuralNetworkModel import NeuralNetworkModel

app = Flask(__name__)

app.add_url_rule('/', view_func=views.index)
# app.add_url_rule('/success/', view_func=views.success, methods=["GET", "POST"])

data ={
    "Company_Name": "Hap Seng Consolidated Sdn Bhd",
    "Report_Year": 2022,
    "Total_Assets": 18654245000,
    "Cash_and_Bank_Balances": 1431980000,
    "Total_Debt": 7069000000,
    "Consolidated_Revenue": 7110496000,
    "Consolidated_Operating_Profit": 1044419000,
    "Cash_to_Total_Assets_Ratio": "7.68%",
    "Debt_to_Total_Assets_Ratio": "37.89%",
    "5_Percent_Benchmark": {
      "Credit_Financing": {
        "Revenue": 241701000,
        "Operating_Profit": 194539000
      }
    },
    "20_Percent_Benchmark": {
        "Property": {
            "Revenue": 523935000,
            "Operating_Profit": 158311000,
        },
        "Trading": {
            "Revenue": 3463801000,
            "Operating_Profit": 297020000,
        }
    },
    "Compliant_Activities": {
        "Plantation": {
            "Revenue": 814554000,
            "Operating_Profit": 266949000
        },
        "Automotive": {
            "Revenue": 1748658000,
            "Operating_Profit": 69284000
        },
        "Building_Materials": {
            "Revenue": 705980000,
            "Operating_Profit": 131770000
        }
    }
}

severeBenchmark = {
    "Credit_Financing": {
        "Revenue": 241701000,
        "Operating_Profit": 194539000
    }
}

lessSevereBenchmark = {
    "Property": {
        "Revenue": 523935000,
        "Operating_Profit": 158311000,
    },
    "Trading": {
        "Revenue": 3463801000,
        "Operating_Profit": 297020000,
    }    
}

@app.route('/success/', methods=["GET", "POST"])   
def success():   
    if request.method == "POST": 
        # r = requests.get('https://jsonplaceholder.typicode.com/posts/1')  
        # data = r.json()
        return render_template("process.html", data = data, severeBench = severeBenchmark, lessSevereBench = lessSevereBenchmark)   
        
if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
    
