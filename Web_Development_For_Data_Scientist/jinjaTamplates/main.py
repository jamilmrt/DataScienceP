from doctest import debug

from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def home():
    name = "jam"
    language= "Python"
    nos = 1,2,3,5,4,7,8,7,9,4,45
    return render_template("index.html", name=name, lang=language, mnos=nos)


app.run(debug=True)