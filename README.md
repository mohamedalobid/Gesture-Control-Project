# Handgesten-Erkennung für Musiksteuerung

Ein Echtzeit-Handgesten-Erkennungssystem, das Computer Vision und Deep Learning verwendet, um die Musikwiedergabe zu steuern. Das System erkennt drei verschiedene Gesten:
- Daumen-Geste: Musik starten
- Handflächen-Geste: Musik stoppen
- Faust-Geste: Musik pausieren

## Funktionen

- Echtzeit-Handgesten-Erkennung mit PyTorch und MediaPipe
- Unterstützung für GPU-Beschleunigung für schnellere Inferenz
- Hochwertige Datenerfassung mit automatischer Qualitätskontrolle
- Umfassende Datenvorverarbeitungs- und Augmentierungs-Pipeline
- Echtzeit-Visualisierung mit Konfidenzwerten
- Unterstützung für Trainings- und Inferenzmodi

## Projektstruktur

```
hand_gesture_recognition/  # Hauptverzeichnis des Projekts
├── data/                  # Verzeichnis für Datensätze
│   ├── raw/                 # Original aufgenommene Bilder
│   ├── processed/           # Vorverarbeitete Bilder
│   └── augmented/           # Erweiterter (augmentierter) Datensatz
├── src/                   # Quellcode-Verzeichnis
│   ├── data_collection/     # Skripte zur Bilderfassung
│   │   └── capture_images.py  # Skript zum Aufnehmen von Handgestenbildern
│   ├── preprocessing/       # Bildvorverarbeitung
│   │   └── preprocess_images.py  # Skript zur Vorverarbeitung der Bilder
│   ├── feature_extraction/  # Merkmalsextraktion
│   │   └── feature_extractor.py  # Skript zur Extraktion von Merkmalen
│   ├── models/             # Modellimplementierung
│   │   └── model.py        # Definition des neuronalen Netzwerkmodells
│   ├── main.py             # Trainingsskript für das Modell
│   └── test_gestures.py    # Skript für Echtzeit-Tests und Gestenerkennung
├── models/
│   └── saved/              # Gespeicherte Modell-Checkpoints (trainierte Modelle)
├── requirements.txt        # Projekt-Abhängigkeiten (erforderliche Python-Pakete)
└── README.md              # Projektdokumentation (diese Datei)
```

## Einrichtung

1. Erstellen Sie eine virtuelle Umgebung:
```bash
python -m venv venv
source venv/bin/activate  # Unter Windows: venv\Scripts\activate
```

2. Installieren Sie die Abhängigkeiten:
```bash
pip install -r requirements.txt
```

## Nutzung

### 1. Datenerfassung

Führen Sie das Datenerfassungsskript aus, um Handgestenbilder aufzunehmen:
```bash
python src/data_collection/capture_images.py
```
Befolgen Sie die Anweisungen auf dem Bildschirm, um Bilder für jede Geste aufzunehmen.

### 2. Training

Trainieren Sie das Modell mit den gesammelten Daten:
```bash
python src/main.py
```
Dies wird:
- Die gesammelten Bilder vorverarbeiten und augmentieren
- Merkmale aus den vorverarbeiteten Bildern extrahieren
- Das PyTorch-Modell mit GPU-Beschleunigung trainieren
- Den besten Modell-Checkpoint speichern

### 3. Testen

Testen Sie das trainierte Modell in Echtzeit:
```bash
python src/test_gestures.py
```
Steuerung:
- Drücken Sie 'q' zum Beenden
- Drücken Sie 'd' zum Umschalten des Debug-Modus (zeigt Konfidenzwerte an)

## Modellarchitektur

Das System verwendet ein PyTorch-basiertes neuronales Netzwerk mit der folgenden Architektur:
- Eingabe: Merkmale von Handgestenbildern
- Versteckte Schichten: 512 → 256 → 128 Neuronen
- Ausgabe: 3 Klassen (Daumen, Handfläche, Faust)
- Aktivierung: ReLU mit Dropout zur Regularisierung
- Verlustfunktion: Kreuzentropie-Verlust (Cross Entropy Loss)
- Optimierer: Adam

## Leistung

Das Modell erreicht:
- Gesamte Genauigkeit: ~77%
- Beste Geste: Faust (92% Recall)
- Echtzeit-Inferenz mit GPU-Beschleunigung

## Anforderungen

- Python 3.8+
- CUDA-fähige GPU (empfohlen für schnellere Verarbeitung)
- Webcam
- Abhängigkeiten, die in requirements.txt aufgeführt sind

## Lizenz

MIT Lizenz

## Mitwirken (Contributing)

1. Forken Sie das Repository
2. Erstellen Sie einen Feature-Branch
3. Committen Sie Ihre Änderungen
4. Pushen Sie zum Branch
5. Erstellen Sie einen Pull Request
