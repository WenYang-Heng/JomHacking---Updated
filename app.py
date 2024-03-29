import views
from flask import Flask, request, render_template

app = Flask(__name__)

app.add_url_rule('/', view_func=views.index)
# app.add_url_rule('/success/', view_func=views.success, methods=["GET", "POST"])

@app.route('/success/', methods=["GET", "POST"])   
def success():   
    if request.method == "POST":   
        return render_template("process.html", data = "something to return")   

    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)


    
