from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd

load_dotenv()

app = Flask(__name__)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = 'your_secret_key'
Session(app)

# Load the CSV file into a pandas DataFrame (do this once)
csv_data = pd.read_csv('medical_data.csv')  # Update with your file path

@app.route('/')
def index():
    if 'history' not in session:
        session['history'] = []
        welcome_message = "I'm here to help! Feel free to ask your questions."
        session['history'].append({'message': welcome_message, 'sender': 'bot'})
    return render_template('index.html', history=session['history'])

@app.route('/submit', methods=['POST'])
def on_submit():
    query = request.form['query']
    session.setdefault('history', []).append({'message': query, 'sender': 'user'})
    
    response = generate_response(query)
    response_message = response
    session['history'].append({'message': response_message, 'sender': 'bot'})
    
    return jsonify({'query': query, 'response': response_message})

def search_csv(query):
    # Search for relevant rows in the CSV
    relevant_rows = csv_data[csv_data['question'].str.contains(query, case=False, na=False)]
    
    if not relevant_rows.empty:
        # Get the most relevant answer (assuming we return the first match)
        return relevant_rows.iloc[0]['answer']
    return None

def generate_response(query):
    # Check for an answer in the CSV file first
    csv_answer = search_csv(query)
    
    if csv_answer:
        # If a relevant answer is found in the CSV, return it
        return csv_answer
    
    # If no relevant answer is found in the CSV, fall back to LLM
    assistant_prompt = "You are a helpful assistant providing medical information based on both LLM knowledge and specific data from trusted medical sources."
    input_text = f"{assistant_prompt}\nUser question:\n{query}"
    assistant = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    result = assistant.invoke(input_text)
    return result.content

if __name__ == '__main__':
    app.run(debug=True)
