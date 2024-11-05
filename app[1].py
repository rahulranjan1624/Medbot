from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer, util
import numpy as np

load_dotenv()

app = Flask(__name__)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = 'your_secret_key'
Session(app)

# Load the CSV file into a pandas DataFrame (do this once)
csv_data = pd.read_csv('medical_data.csv')  # Update with your file path

# Load NLP models
nlp = spacy.load("en_core_web_sm")  # SpaCy model for basic NLP tasks
model = SentenceTransformer('all-MiniLM-L6-v2')  # Sentence Transformer for semantic similarity

# Utility function to encode sentences using Sentence Transformers
def encode_sentences(sentences):
    # Returns a list of 1D arrays (sentence embeddings)
    return [model.encode(sentence, convert_to_tensor=True) for sentence in sentences]

# Pre-encode the questions from the CSV for efficient similarity checking
csv_data['question_embedding'] = encode_sentences(csv_data['question'].values)

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

def search_csv_semantically(query, top_n=1):
    """
    Search for the most relevant row in the CSV using semantic similarity with Sentence Transformers.
    Returns the top N matching rows.
    """
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = [util.pytorch_cos_sim(query_embedding, emb)[0][0].item() for emb in csv_data['question_embedding']]
    
    # Get the top N most similar results
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    top_rows = csv_data.iloc[top_indices]
    
    return top_rows, np.array(similarities)[top_indices]

def generate_response(query):
    """
    Generate a response using both CSV data and the LLM. It first tries to find a match in the CSV,
    and then queries the LLM for additional context or if the CSV match is weak.
    """
    # Step 1: Find the best match in the CSV based on semantic similarity
    top_rows, top_similarities = search_csv_semantically(query, top_n=3)
    
    # Step 2: Set a similarity threshold for trusting the CSV answer
    similarity_threshold = 0.7
    csv_answer = None
    csv_source = None
    csv_focus_area = None
    
    # If a match is found with high similarity, use it as the main response
    if top_similarities[0] >= similarity_threshold:
        csv_answer = top_rows.iloc[0]['answer']
        csv_source = top_rows.iloc[0]['source']
        csv_focus_area = top_rows.iloc[0]['focus_area']
    
    # Step 3: If no strong match is found in the CSV, fallback to LLM
    if csv_answer is None:
        assistant_prompt = "You are a medical assistant helping with health-related questions. You must provide relevant, trusted information."
        input_text = f"{assistant_prompt}\nUser question:\n{query}"
        assistant = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        llm_result = assistant.invoke(input_text)
        return llm_result.content
    else:
        # Combine CSV answer with LLM context if the match was weaker
        assistant_prompt = f"You are a medical assistant with knowledge in {csv_focus_area}. Provide additional context based on the user's question."
        input_text = f"{assistant_prompt}\nUser question:\n{query}\nCurrent Answer: {csv_answer}"
        assistant = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        llm_result = assistant.invoke(input_text)
        
        # Step 4: Merge CSV data and LLM-generated response
        combined_response = f"{csv_answer}\n\nAdditional Context from AI: {llm_result.content}\n\nSource: {csv_source}"
        return combined_response

if __name__ == '__main__':
    app.run(debug=True)
