from flask import Flask , jsonify , request
from langchain_google_genai import ChatGoogleGenerativeAI
from api import GEMINI_API_KEY
from utils import docs_loader , text_splitter , download_hugging_face_embeddings , propmt_completion , creat_db

urls = ['https://www.victoriaonmove.com.au/local-removalists.html' , 'https://www.victoriaonmove.com.au/index.html' , 'https://www.victoriaonmove.com.au/contact.html']
data = docs_loader(urls=urls)
chunks = text_splitter(data = data)
embeddings = download_hugging_face_embeddings()
retriever = creat_db(chunks= chunks , embeddings= embeddings)
llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash" , api_key = GEMINI_API_KEY)



app = Flask(__name__)


@app.route('/generate_response')
def index():
    data = request.get_json()
    message = data['message']
    answer = propmt_completion(llm , message , retriever)
    return jsonify({"Answer" : f"{answer}"})



if __name__ == '__main__':
    app.run(debug = True)