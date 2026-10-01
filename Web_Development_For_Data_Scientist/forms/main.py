from doctest import debug

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        print(request.form)
        name = request.form['name']
        password = request.form['password']
        print(f"The name is {name} and Password is {password} ")
        return "<b> Thanks for using  </b>"
    return render_template("index.html")


app.run(debug=True)