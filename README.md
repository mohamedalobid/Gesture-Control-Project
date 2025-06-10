# Handgesten-Erkennung für Musiksteuerung

Ein Echtzeit-Handgesten-Erkennungssystem, das Computer Vision und Deep Learning verwendet, um die Musikwiedergabe zu steuern. Das System erkennt drei verschiedene Gesten:
- Daumen-Geste: Musik starten
- Handflächen-Geste: Musik stoppen
- Faust-Geste: Musik pausieren

## Funktionen

- Echtzeit-Handgesten-Erkennung mit PyTorch und MediaPipe
- Verbesserte Datenerfassung mit Echtzeit-Visualisierung, Handpositionierung und Qualitätsprüfung (grüner/roter Rahmen)
- Fortschrittliche Merkmalsextraktion (HOG, formbasiert, statistisch)
- Unterstützung für GPU-Beschleunigung für schnellere Inferenz und Training
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
│   │   └── capture_images.py  # Skript zum Aufnehmen von Handgestenbildern (mit visuellem Feedback)
│   ├── preprocessing/       # Bildvorverarbeitung
│   │   └── preprocess_images.py  # Skript zur Vorverarbeitung der Bilder
│   ├── feature_extraction/  # Merkmalsextraktion
│   │   └── feature_extractor.py  # Skript zur Extraktion von HOG-, Form- und statistischen Merkmalen
│   ├── models/             # Modellimplementierung
│   │   └── model.py        # Definition des neuronalen Netzwerkmodells (FFNN)
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
Befolgen Sie die Anweisungen auf dem Bildschirm. Die Anwendung zeigt einen grünen Rahmen um Ihre Hand, wenn die Position gut ist, und einen roten Rahmen, wenn sie angepasst werden muss. Drücken Sie 'c' zum Aufnehmen von Bildern, wenn der Rahmen grün ist.

### 2. Training

Trainieren Sie das Modell mit den gesammelten Daten:
```bash
python src/main.py
```
Dies wird:
- Die gesammelten Bilder vorverarbeiten und augmentieren
- HOG-, Form- und statistische Merkmale aus den vorverarbeiteten Bildern extrahieren
- Das PyTorch-FFNN-Modell mit GPU-Beschleunigung trainieren, mit Early Stopping und Lernraten-Scheduler
- Den besten Modell-Checkpoint speichern

### 3. Testen

Testen Sie das trainierte Modell in Echtzeit:
```bash
python src/test_gestures.py
```
Das Skript zeigt die erkannte Geste, die zugehörige Aktion (z.B. "Start Musik") und die Konfidenzwerte in Echtzeit an.
Steuerung:
- Drücken Sie 'q' zum Beenden

## Modellarchitektur

Das System verwendet ein PyTorch-basiertes Feed Forward Neuronales Netzwerk (FFNN) mit der folgenden Architektur:
- Eingabe: Extrahierte Merkmale (HOG, Form, statistisch)
- Versteckte Schichten: 1024 → 512 → 256 Neuronen (mit ReLU-Aktivierung, Batch-Normalisierung und Dropout)
- Ausgabe: 3 Klassen (Daumen, Handfläche, Faust)
- Verlustfunktion: Kreuzentropie-Verlust (Cross Entropy Loss)
- Optimierer: Adam
- Techniken: Lernraten-Scheduler (ReduceLROnPlateau) und Early Stopping

## Leistung

Das Modell erreicht:
- Gesamte Genauigkeit: ~77% (Bitte beachten Sie, dass dies ein Beispielwert ist. Die tatsächliche Leistung kann variieren.)
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
