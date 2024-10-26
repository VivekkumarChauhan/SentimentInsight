# Base Python image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download NLTK data resources needed for sentiment analysis
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('vader_lexicon')"

# Download SpaCy language models
RUN python -m spacy download en_core_web_sm

# Expose the port for the Flask API
EXPOSE 5000

# Start the Flask API
CMD ["python", "app/api.py"]
