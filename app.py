import streamlit as st
import whisper
import arabic_reshaper
from bidi.algorithm import get_display
import difflib
import Levenshtein

# The reference surah
REFERENCE_SURAH = "انا اعطيناك الكوثر فصل لربك وانحر ان شانئك هو الابتر"

# Define a dictionary with common transcription errors and their corrections
import re

# Define a dictionary with common transcription errors and their corrections
COMMON_ERRORS = {
    "إنها": "انا",
    "أقفينا": "اعطيناك",
    "افضن": "اعطيناك",
    "كريك": "الكوثر",
    "كركو": "الكوثر",
    "سر": "فصل",
    "فصل لرنبك": "فصل لربك",
    "لرنبك": "لربك",
    "كرن حرق": "وانحر",
    "بكا": "لربك",
    "شانئ أكه": "شانئك",
    "ورئ": "هو",
    "ورأذتو": "الابتر",
    "أذن": "الابتر",
    "إن":"انا",
    "ألقى":"الكو",
    "ألسر":"ثر",
    "ألقى ألسر":"الكوثر",
    "أفوين":"اعطيناك",
    "غلقوصر":" الكوثر",
    "أقفين":"اعطيناك",
    "كرك وصر":"الكوثر",
    "بقل لربك":"فصل لربك",
    "بقل":"فصل",
    "ونحر":"وانحر",
    "شانئك":"شانئك",
    "هو":"هو",
    "الابتر":"الابتر",
    "بس من آذ وحمن وحيب": "بسم الله الرحمن الرحيم",
    "بس":"بسم",
    "وحيب":" الرحيم",
    "وحمن":"الرحمن",
    "انا شانئك":"ان شانئك",
    "هو الابتر":"هو الابتر",
    "عنها":"وانحر",
    "الأبتر":"الابتر",
    "إن":" ان",
    "عنها":"وانحر",
    "شانعك":"شانئك",
    "الكوسر":"الكوثر",
    "ونهر":" وانحر",
    " الأبثر":" الابتر"
    
}


def remove_non_arabic(text):
    """
    Removes non-Arabic characters and symbols from the transcribed text.
    """
    # Use a regex to keep only Arabic letters and spaces
    arabic_text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    return arabic_text


def find_closest_match(word, error_dict, max_distance=2):
    closest_word = None
    closest_distance = float('inf')

    for wrong_word, correct_word in error_dict.items():
        distance = Levenshtein.distance(word, wrong_word)
        if distance < closest_distance and distance <= max_distance:
            closest_word = correct_word
            closest_distance = distance

    return closest_word
def correct_transcription(transcribed_text):
    """
    Preprocesses and corrects the transcribed text by matching it with known common errors.
    """
    # Step 1: Remove non-Arabic characters from the transcription
    transcribed_text = remove_non_arabic(transcribed_text)

    # Split the transcribed text into words
    transcribed_words = transcribed_text.split()

    # Initialize an empty list to hold corrected words
    corrected_words = []

    i = 0
    while i < len(transcribed_words):
        current_word = transcribed_words[i]

        # Check if the current word combined with the next one forms a known error
        if i < len(transcribed_words) - 1:
            combined_words = current_word + " " + transcribed_words[i + 1]
            if combined_words in COMMON_ERRORS:
                corrected_word = COMMON_ERRORS[combined_words]
                corrected_words.append(corrected_word)
                i += 2  # Skip the next word since we handled it as part of the combined phrase
                continue

        # If the word exists in the COMMON_ERRORS dictionary, replace it
        if current_word in COMMON_ERRORS:
            corrected_word = COMMON_ERRORS[current_word]
        else:
            # If no direct match, retain the original word
            corrected_word = current_word

        # Append the corrected word to the list
        corrected_words.append(corrected_word)
        i += 1

    # Join the corrected words back into a single string
    corrected_text = " ".join(corrected_words)

    return corrected_text


import whisper
import os



def transcribe_audio(file_path, model_name="medium"):
    """Transcribes the given audio file and returns the transcribed text using OpenAI Whisper."""
    try:
        # Since the model is in the current directory, no need to specify a separate models folder
        model_path = f"{model_name}.pt"

        # Load the Whisper model from the current directory
        model = whisper.load_model(model_path)
        
        # Transcribe audio using Whisper
        result = model.transcribe(file_path, language='ar')

        return result['text'].strip()  # Return the transcribed text
    except Exception as e:
        return f"حدث خطأ: {str(e)}"


def compare_texts(reference, transcription):
    """Compares reference text with transcription and returns colored HTML."""
    reference_words = reference.strip().split()
    transcription_words = transcription.strip().split()

    colored_output = []
    ref_index = 0  # Track position in the reference string

    # Iterate through transcription words
    for trans_word in transcription_words:
        # Check if we have reached the end of the reference
        if ref_index < len(reference_words):
            if trans_word == reference_words[ref_index]:
                # If the word matches, color it green and move to the next word in the reference
                colored_output.append(f"<span style='color:green;'>{trans_word}</span>")
                ref_index += 1  # Move to the next word in the reference
            else:
                # If it does not match, color it red
                colored_output.append(f"<span style='color:red;'>{trans_word}</span>")
        else:
            # If we've exhausted the reference words, color remaining transcription words red
            colored_output.append(f"<span style='color:red;'>{trans_word}</span>")

    return " ".join(colored_output)

import difflib

import re
import difflib

def remove_arabic_diacritics(text):
    """Remove Arabic diacritics (tashkeel) to normalize text for comparison."""
    arabic_diacritics = re.compile("""
        ّ    | # Tashdid
        َ    | # Fatha
        ً    | # Tanwin Fath
        ُ    | # Damma
        ٌ    | # Tanwin Damm
        ِ    | # Kasra
        ٍ    | # Tanwin Kasr
        ْ    | # Sukun
        ـ     # Tatwil/Kashida
    """, re.VERBOSE)
    # Remove the diacritics
    text = re.sub(arabic_diacritics, '', text)
    return text

def normalize_arabic_text(text):
    """Normalize Arabic text for comparison."""
    text = remove_arabic_diacritics(text)
    # Replace Hamza forms, etc. (You can add more normalization rules here)
    text = text.replace('إ', 'ا').replace('أ', 'ا').replace('آ', 'ا').replace('ة', 'ه')
    return text

def highlight_transcription(reference, transcription):
    """Highlights the transcription with green for correct matches and red for incorrect, using fuzzy matching and normalization."""
    # Normalize both the reference and transcription
    normalized_reference = normalize_arabic_text(reference)
    normalized_transcription = normalize_arabic_text(transcription)
    
    matcher = difflib.SequenceMatcher(None, normalized_reference.split(), normalized_transcription.split())
    highlighted_output = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Highlight matched (equal) words in green
            matched_part = " ".join(transcription.split()[j1:j2])
            highlighted_output.append(f"<span style='color:green;'>{matched_part}</span>")
        else:
            # Highlight non-matching parts in red
            mismatched_part = " ".join(transcription.split()[j1:j2])
            highlighted_output.append(f"<span style='color:red;'>{mismatched_part}</span>")

    return " ".join(highlighted_output)

def highlight_correction(reference, transcription):
    """Highlights the reference with green for correct parts based on the transcription using fuzzy matching and normalization."""
    # Normalize both the reference and transcription
    normalized_reference = normalize_arabic_text(reference)
    normalized_transcription = normalize_arabic_text(transcription)

    matcher = difflib.SequenceMatcher(None, normalized_reference.split(), normalized_transcription.split())
    highlighted_output = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Highlight matched (equal) parts in green
            matched_part = " ".join(reference.split()[i1:i2])
            highlighted_output.append(f"<span style='color:green;'>{matched_part}</span>")
        else:
            # Leave the rest white
            unmatched_part = " ".join(reference.split()[i1:i2])
            highlighted_output.append(f"<span style='color:white;'>{unmatched_part}</span>")

    return " ".join(highlighted_output)




def main():
    # Streamlit app title
    st.title("Quran Speech to Text")

    # Add custom CSS for Arabic font
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Amiri&display=swap');

            .arabic-text {
                font-family: 'Amiri', serif;  /* Use a nice Arabic font */
                font-size: 28px;              /* Increased font size for better readability */
                direction: rtl;               /* Right to left direction */
                text-align: right;            /* Align text to right */
                line-height: 1.6;             /* Increase line height for better spacing */
            }
        </style>
        """, unsafe_allow_html=True)

    # File uploader for audio files
    uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "flac", "ogg", "aac", "wma"])
    
    # Initialize a variable to store the audio file path
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with open("temp_audio_file", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Create a button to proceed with transcription
        if st.button("Proceed Audio"):
            # Transcribe the audio file
            transcribed_text = transcribe_audio("D:\\vscode\\quran-bot\\quran-bot\\temp_audio_file")
            
            # Correct the transcribed text using the preprocessing function
            corrected_transcribed_text = correct_transcription(transcribed_text)
            
            # Reshape and display the text
            reshaped_text = arabic_reshaper.reshape(corrected_transcribed_text)
            bidi_text = get_display(reshaped_text)

            # Compare transcribed text with the reference surah and get colored output
            comparison_html = compare_texts(REFERENCE_SURAH, corrected_transcribed_text)

            # Display the result with mismatches highlighted

            # Highlight mistakes in the reference string and display corrections
# Display the original transcription section
# Display the original transcription section
            st.header("Transcription")
            transcription_html = highlight_transcription(REFERENCE_SURAH, transcribed_text)
            st.markdown(f"<p class='arabic-text'>{transcription_html}</p>", unsafe_allow_html=True)

            # Display the corrected section with the original Surah
            st.header("Correction")
            correction_html = highlight_correction(REFERENCE_SURAH, transcribed_text)
            st.markdown(f"<p class='arabic-text'>كيف تقولها: {correction_html}</p>", unsafe_allow_html=True)





if __name__ == "__main__":
    main()