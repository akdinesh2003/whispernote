import streamlit as st
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import tempfile
import librosa
import soundfile as sf
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

class WhisperNoteApp:
    def __init__(self):
        self.model = whisper.load_model("base")
        self.summarization_model_name = "facebook/bart-base"
        self.summarization_tokenizer = AutoTokenizer.from_pretrained(self.summarization_model_name)
        self.summarization_model = AutoModelForCausalLM.from_pretrained(self.summarization_model_name)
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def transcribe_audio(self, audio_file_path):
        try:
            y, sr = librosa.load(audio_file_path, sr=16000)
            result = self.model.transcribe(y)
            return result["text"]
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            return None

    def summarize_text(self, text):
        input_ids = self.summarization_tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
        attention_mask = self.summarization_tokenizer.encode(text, return_tensors='pt', max_length=512, padding='max_length', truncation=True)

        output = self.summarization_model.generate(input_ids, attention_mask=attention_mask, max_length=150, early_stopping=True)
        summary = self.summarization_tokenizer.decode(output[0], skip_special_tokens=True)
        return summary

    def tag_and_categorize(self, text):
        # Enhanced tagging and categorization using nltk
        words = word_tokenize(text)
        words = [self.lemmatizer.lemmatize(w.lower()) for w in words if w.isalpha() and w not in self.stop_words]

        tags = []
        categories = []

        keywords = {
            "meeting": {"tags": ["Meeting Notes"], "categories": ["Business"]},
            "lecture": {"tags": ["Lecture Notes"], "categories": ["Education"]},
            "project": {"tags": ["Project Management"], "categories": ["Business"]},
            "research": {"tags": ["Research"], "categories": ["Education", "Science"]},
        }

        for word in words:
            if word in keywords:
                tags.extend(keywords[word]["tags"])
                categories.extend(keywords[word]["categories"])

        return {"tags": list(set(tags)), "categories": list(set(categories))}

    def integrate_with_note_taking_app(self, text, summary, tags, categories):
        # Placeholder for integration with note-taking apps
        st.write("\n--- Integration with Note-Taking App ---")
        st.write(f"Saving to note-taking app:\nText: {text}\nSummary: {summary}\nTags: {tags}\nCategories: {categories}")
        st.write("--- End of Integration ---\n")


def main():
    st.title("WhisperNote App")
    st.write("Upload an audio file to transcribe and summarize.")

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])

    if uploaded_file is not None:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            audio_file = tmp_file.name  # Get the temporary file path

        st.write(f"Using temporary file: {audio_file}")

        # Load and resave the audio file using librosa/soundfile
        try:
            y, sr = librosa.load(audio_file, sr=16000)  # Load with 16kHz sampling rate
            sf.write(audio_file, y, 16000, format='WAV', subtype='PCM_16')  # Save as 16kHz WAV
            st.write("Audio file resaved using librosa/soundfile.")
        except Exception as e:
            st.error(f"Error processing audio with librosa/soundfile: {e}")
            os.remove(audio_file)
            st.write("Temporary file removed.")
            return

        app = WhisperNoteApp()
        text = app.transcribe_audio(audio_file)

        if text:
            st.header("Transcribed Text")
            st.write(text)

            summary = app.summarize_text(text)
            st.header("Summary")
            st.write(summary)

            tags_categories = app.tag_and_categorize(text)
            st.header("Tags and Categories")
            st.write(tags_categories)

            app.integrate_with_note_taking_app(text, summary, tags_categories["tags"], tags_categories["categories"])

        # Clean up the temporary file
        os.remove(audio_file)
        st.write("Temporary file removed.")

if __name__ == "__main__":
    main()