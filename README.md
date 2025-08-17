# WhisperNote

A voice-to-text note-taker application built with Streamlit, OpenAI's Whisper, Transformers, and NLTK. This application allows you to upload audio files, transcribe them into text, summarize the text, and categorize the content.

## Features

*   **Voice-to-Text Transcription:** Accurately transcribes audio files using OpenAI's Whisper ASR model.
*   **Text Summarization:** Generates concise summaries of transcribed text using the Transformers library.
*   **Tagging and Categorization:** Automatically tags and categorizes notes using NLTK for NLP.
*   **Web Interface:** User-friendly web interface built with Streamlit.
*   **Easy Deployment:** Simple instructions for local deployment.
*   **Multiple Audio Formats:** Supports various audio formats like WAV, MP3, and M4A.

## Technologies Used

*   [Streamlit](https://streamlit.io/): For creating the web interface.
*   [OpenAI Whisper](https://openai.com/blog/whisper/): For voice-to-text transcription.
*   [Transformers](https://huggingface.co/transformers/): For text summarization.
*   [NLTK](https://www.nltk.org/): For natural language processing (tagging and categorization).
*   [Librosa](https://librosa.org/): For audio loading and manipulation.
*   [SoundFile](http://www.mega-nerd.com/libsndfile/): For saving audio files.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/akdinesh2003/whispernote.git
    cd whispernote
    ```

2.  **Install the dependencies:**

    ```bash
    pip install streamlit openai-whisper transformers nltk librosa soundfile
    ```

3.  **Download NLTK data:**
    Open a python shell and run the following

    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    ```

4.  **Run the application:**

    ```bash
    streamlit run app.py
    ```

    Open your web browser and go to the address provided by Streamlit (usually `http://localhost:8501`).

## Usage

1.  **Open the WhisperNote app** in your web browser.
2.  **Upload an audio file** using the "Choose an audio file" button. Supported formats are WAV, MP3, and M4A.
3.  **Wait for the processing to complete.** The application will transcribe the audio, summarize the text, and display the tags and categories.

## Deployment

To deploy this application, you can use various platforms. Here are a few options:

### 1. Heroku

*   Sign up for a Heroku account.
*   Install the Heroku CLI.
*   Log in to Heroku: `heroku login`
*   Create a Heroku app: `heroku create`
*   Deploy the app:

    ```bash
    git init
    git add .
    git commit -m "Deploy to Heroku"
    heroku git:remote -a <your-heroku-app-name>
    git push heroku main
    ```

### 2. Streamlit Cloud

*   Create a Streamlit account.
*   Connect your GitHub repository to Streamlit Cloud.
*   Streamlit Cloud will automatically deploy your application.

### 3. Google Cloud

*   Create a Google Cloud account.
*   Use Google Cloud App Engine or Cloud Run to deploy the application.

## Notes

*   The first time you run the application, it may take some time to download the Whisper model.
*   The accuracy of the transcription depends on the quality of the audio file.
*   The summarization and tagging features are based on NLP techniques and may not be perfect.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Author

*   AK DINESH - [https://github.com/akdinesh2003](https://github.com/akdinesh2003)

