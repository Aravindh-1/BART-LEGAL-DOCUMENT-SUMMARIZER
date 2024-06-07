from flask import Flask, render_template, request
from transformers import BartForConditionalGeneration, BartTokenizer
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import docx2txt

app = Flask(__name__)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the BART model and tokenizer
model_path = "LDS Model"  # Adjust this path to the directory where your model is stored
model = BartForConditionalGeneration.from_pretrained(model_path)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Define NLTK preprocessing function
def preprocess_text(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Optionally, remove stopwords
    stop_words = set(stopwords.words('english'))
    preprocessed_sentences = []
    for sentence in sentences:
        words = sentence.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        preprocessed_sentences.append(' '.join(filtered_words))
    
    # Optionally, perform stemming
    stemmer = PorterStemmer()
    stemmed_sentences = [stemmer.stem(sentence) for sentence in preprocessed_sentences]
    
    # Join the preprocessed sentences back into a single string
    preprocessed_text = ' '.join(stemmed_sentences)
    
    return preprocessed_text

# Define function for text summarization
def generate_summary(text):
    preprocessed_text = preprocess_text(text)
    input_ids = tokenizer(preprocessed_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(input_ids["input_ids"], num_beams=4, max_length=512, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', message='No selected file')
    
    # Read the file
    if file.filename.endswith('.txt'):
        text = file.read().decode("utf-8")
    elif file.filename.endswith('.pdf'):
        # Add PDF processing logic here
        pass
    elif file.filename.endswith('.docx'):
        text = docx2txt.process(file)

    # Generate summary
    summary = generate_summary(text)

    # Display summary
    return render_template('index.html', message='File uploaded successfully', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)



# from flask import Flask, render_template, request

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return render_template('index.html', message='No file part')
    
#     file = request.files['file']
    
#     if file.filename == '':
#         return render_template('index.html', message='No selected file')
    
#     # Here you can process the uploaded file, save it, or do whatever you want with it
#     # For example, save the file to a specific directory
#     file.save('uploads/' + file.filename)
    
#     message = 'File uploaded successfully'
#     return render_template('index.html', message=message)

# if __name__ == '__main__':
#     app.run(debug=True)
