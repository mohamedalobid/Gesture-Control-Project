import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from preprocessing.preprocess_images import ImagePreprocessor
from models.model import GestureClassifier

# Funktion zum Laden und Vorbereiten des Datensatzes für das Training
def load_dataset(data_dir):
    """Lädt und bereitet den Datensatz für das Training vor"""
    X = []  # Liste zur Speicherung der Bilddaten
    y = []  # Liste zur Speicherung der Labels (Gesten)
    gesture_map = {'thumb': 0, 'palm': 1, 'fist': 2}  # Abbildung von Gestennamen auf numerische Labels
    
    print("Datensatz wird geladen...")
    # Iteration durch jeden Gestenordner im Datenverzeichnis
    for gesture in os.listdir(data_dir):
        gesture_dir = os.path.join(data_dir, gesture)  # Pfad zum aktuellen Gestenordner
        # Überspringen, wenn es kein Verzeichnis ist
        if not os.path.isdir(gesture_dir):
            continue
            
        print(f"Lade {gesture}-Gesten...")
        # Iteration durch jede Bilddatei im Gestenordner
        for img_name in os.listdir(gesture_dir):
            # Überspringen, wenn es keine Bilddatei ist
            if not img_name.endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            img_path = os.path.join(gesture_dir, img_name)  # Vollständiger Pfad zum Bild
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Bild als Graustufen laden
            
            # Überspringen, wenn das Bild nicht geladen werden konnte
            if image is None:
                continue
            
            # Normalisiere Pixelwerte auf den Bereich [0, 1]
            image = image.astype(np.float32) / 255.0
            X.append(image)  # Bild zu den Feature-Daten hinzufügen
            y.append(gesture_map[gesture])  # Label zur Liste der Labels hinzufügen
    
    return np.array(X), np.array(y)  # Rückgabe der Bilddaten und Labels als NumPy-Arrays

# Hauptfunktion des Programms
def main():
    # Erstelle notwendige Verzeichnisse, falls sie nicht existieren
    os.makedirs('Gesture Control/data/raw', exist_ok=True)  # Rohe Bilddaten
    os.makedirs('Gesture Control/data/processed', exist_ok=True)  # Verarbeitete Bilddaten
    os.makedirs('Gesture Control/models/saved', exist_ok=True)  # Speicherort für trainierte Modelle
    
    # Initialisiere und führe den Bildvorverarbeiter aus
    preprocessor = ImagePreprocessor()  # Erstelle eine Instanz des Bildvorverarbeiters
    preprocessor.preprocess_dataset()  # Vorverarbeite den Datensatz
    preprocessor.augment_dataset()  # Erweitere den Datensatz (Data Augmentation)
    
    # Lade den vorverarbeiteten Datensatz
    X, y = load_dataset('Gesture Control/data/processed')  # Lade die vorverarbeiteten Bilder und Labels
    
    # Überprüfen, ob Daten gefunden wurden
    if len(X) == 0:
        print("Keine Daten gefunden! Bitte sammeln Sie zuerst Trainingsdaten.")
        return  # Programm beenden, wenn keine Daten vorhanden sind
    
    print(f"\n{len(X)} Bilder geladen")
    print(f"Klassenverteilung: {np.bincount(y)}")  # Zeigt die Anzahl der Bilder pro Klasse
    
    # Teile die Daten in Trainings- und Validierungssets auf
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) # 80% Training, 20% Validierung
    
    # Initialisiere und trainiere das Modell
    model = GestureClassifier()  # Erstelle eine Instanz des Gestenklassifizierers
    model.train_model(X_train, y_train, batch_size=32, epochs=200) # Trainiere das Modell mit den Trainingsdaten
    
    # Speichere das beste Modell
    model.save_model('Gesture Control/models/saved/best_model')  # Speichere das trainierte Modell
    print("\nModell erfolgreich gespeichert!")

# Überprüfen, ob das Skript direkt ausgeführt wird
if __name__ == "__main__":
    main()  # Rufe die Hauptfunktion auf, um das Programm zu starten 