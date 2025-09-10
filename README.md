# YouTube Transcript Summarizer (Deutsch)

Dieses Projekt lädt automatisch Untertitel von YouTube-Videos herunter, erstellt eine kurze Zusammenfassung mit einem vortrainierten **BART-Modell** und übersetzt diese ins **Deutsche**.

## 🚀 Features
- Automatisches Abrufen von YouTube-Untertiteln (manuell oder automatisch generiert).  
- Bereinigung und Chunking langer Transkripte.  
- Erstellung einer englischen Zusammenfassung mit **facebook/bart-large-cnn**.  
- Übersetzung der Zusammenfassung ins Deutsche mit **Helsinki-NLP/opus-mt-en-de**.  
- Unterstützung für GPU-Beschleunigung (CUDA) – fällt automatisch auf CPU zurück, falls keine GPU verfügbar ist.  

## 📦 Installation

### Voraussetzungen
- Python 3.8 oder neuer  
- pip Paketmanager  
- (Optional) CUDA-fähige GPU  

### Schritte
1. Repository klonen:
```bash
git clone https://github.com/DEIN-USERNAME/youtube-summarizer.git
cd youtube-summarizer
