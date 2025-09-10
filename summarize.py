# Installiere benötigte Bibliotheken vor dem Ausführen dieses Skripts:
!pip install youtube_transcript_api transformers datasets

import os
import re
import torch
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer

# Setze CUDA_LAUNCH_BLOCKING für synchrones Debugging (hilfreich bei CUDA-Fehlern)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Versuche, das Gerät zu verwenden: GPU, wenn verfügbar, ansonsten CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Versuche, das Gerät zu verwenden:", device)

#############################################
# 1. Eingabe der YouTube-Video-URL
#############################################
video_url = input("Bitte gib die YouTube Video URL ein: ")

# Extrahiere die Video-ID aus der URL
match = re.search(r"(?:v=|youtu\.be/)([\w-]+)", video_url)
if match:
    video_id = match.group(1)
else:
    raise ValueError("Ungültige YouTube URL.")

#############################################
# 2. Untertitel abrufen (manuell oder automatisch)
#############################################
try:
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    transcript = None
    # Versuche, einen manuell erstellten Untertitel zu finden
    for t in transcript_list:
        if not t.is_generated:
            transcript = t.fetch()
            print(f"Manuell erstellter Untertitel gefunden in der Sprache: {t.language}")
            break

    # Falls kein manueller Untertitel vorhanden ist, verwende den automatisch generierten
    if transcript is None:
        # Hier verwenden wir t.language_code statt t.language
        auto_transcript = transcript_list.find_generated_transcript([t.language_code for t in transcript_list])
        transcript = auto_transcript.fetch()
        print(f"Automatisch generierter Untertitel wird genutzt in der Sprache: {auto_transcript.language}")
except Exception as e:
    print("Fehler beim Abrufen der Untertitel:", e)
    raise SystemExit("Es konnten keine Untertitel abgerufen werden.")

# Kombiniere alle Untertitel-Teile in einen langen String
full_text = " ".join([entry["text"] for entry in transcript])
print("Länge des kombinierten Untertiteltextes:", len(full_text))

# Optional: Textbereinigung (z. B. Entfernen nicht druckbarer Zeichen)
def clean_text(text):
    return re.sub(r'[\x00-\x1f]+', ' ', text)

full_text = clean_text(full_text)

#############################################
# 3. Zusammenfassung des Untertiteltextes
#############################################
model_name = "facebook/bart-large-cnn"

# Lade das Modell und den Tokenizer
model = BartForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Versuche, das Modell auf das gewünschte Gerät zu verschieben.
# Falls dabei ein Fehler auftritt, wird automatisch auf CPU zurückgefallen.
try:
    model.to(device)
    print("Modell erfolgreich auf", device, "verschoben.")
except RuntimeError as e:
    print("Fehler beim Verschieben des Modells auf das Gerät:", e)
    print("Falle auf CPU zurück.")
    device = torch.device("cpu")
    model.to(device)

# Erstelle die Pipeline. Hier muss als device-Parameter ein Index übergeben werden:
# Verwende 0, wenn die GPU genutzt wird, ansonsten -1 für CPU.
pipeline_device = 0 if device.type == "cuda" else -1
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=pipeline_device)

# Funktion: Zerlege den Text in Chunks, die das maximale Token-Limit nicht überschreiten
def chunk_text(text, max_tokens, tokenizer):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk = tokenizer.decode(chunk_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        chunks.append(chunk)
    return chunks

# Bestimme die maximale Eingabelänge (z. B. 1024 Token)
max_input_length = tokenizer.model_max_length
chunks = chunk_text(full_text, max_input_length, tokenizer)
print(f"Text wurde in {len(chunks)} Chunk(s) aufgeteilt.")

# Fasse die einzelnen Chunks in einer Schleife zusammen (ohne Batch-Verarbeitung, um GPU-Fehler zu vermeiden)
chunk_summaries = []
for idx, chunk in enumerate(chunks):
    print(f"Verarbeite Chunk {idx+1} von {len(chunks)} ...")
    summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False, truncation=True)
    chunk_summaries.append(summary[0]["summary_text"])

# Falls mehr als ein Chunk vorliegt, fassen wir die einzelnen Zusammenfassungen nochmals zusammen
if len(chunk_summaries) > 1:
    combined_summary_text = " ".join(chunk_summaries)
    print("Fasse die kombinierten Zusammenfassungen nochmals zusammen ...")
    final_summary = summarizer(combined_summary_text, max_length=130, min_length=30, do_sample=False, truncation=True)[0]["summary_text"]
else:
    final_summary = chunk_summaries[0]

print("\nZusammenfassung (vor Übersetzung):")
print(final_summary)

#############################################
# 4. Übersetzung der Zusammenfassung ins Deutsche
#############################################
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de", device=pipeline_device)
translated = translator(final_summary, max_length=512)
translated_summary = translated[0]["translation_text"]

#############################################
# 5. Ausgabe der finalen Zusammenfassung
#############################################
print("\nFinale Zusammenfassung (auf Deutsch):")
print(translated_summary)
