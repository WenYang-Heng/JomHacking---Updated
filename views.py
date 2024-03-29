from flask import render_template, request
from fileinput import filename

def index():
    return render_template('index.html')

# def success():
#     if request.method == 'POST':
#         f = request.files['file'] 
#         f.save(f.filename)  
#         return render_template("process.html")
