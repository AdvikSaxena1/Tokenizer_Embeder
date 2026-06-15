from flask import Flask, render_template, request

from tokenizer import tokenize
from embedding import generate_token_embeddings

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():

    embeddings = []
    user_text = ""

    if request.method == "POST":

        user_text = request.form["text"]

        tokens = tokenize(user_text)

        embeddings = generate_token_embeddings(tokens)

    return render_template(
        "index.html",
        embeddings=embeddings,
        user_text=user_text
    )

if __name__ == "__main__":
    app.run(debug=True)