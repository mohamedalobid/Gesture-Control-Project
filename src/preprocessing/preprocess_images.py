import cv2
import numpy as np
import os
from pathlib import Path
import mediapipe as mp

# Definition der Klasse ImagePreprocessor
class ImagePreprocessor:
    # Konstruktor der Klasse
    def __init__(self, input_dir='Gesture Control/data/raw', processed_dir='Gesture Control/data/processed', augmented_dir='Gesture Control/data/augmented'):
        self.input_dir = input_dir  # Verzeichnis für rohe Bilder
        self.processed_dir = processed_dir  # Verzeichnis für vorverarbeitete Bilder
        self.augmented_dir = augmented_dir  # Verzeichnis für augmentierte Bilder
        # Initialisiere das Hand-Erkennungsmodell (MediaPipe Hands)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Erstelle Ausgabeverzeichnisse, falls sie nicht existieren
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(augmented_dir, exist_ok=True)
    
    # Funktion zur Erkennung einer Hand in einem Bild
    def detect_hand(self, image):
        """Detect hand in the image using MediaPipe"""
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Bild von BGR zu RGB konvertieren
        results = self.hands.process(rgb)  # Hand-Landmarken mit MediaPipe verarbeiten
        
        if results.multi_hand_landmarks:  # Wenn Hand-Landmarken erkannt wurden
            # Wähle die erste erkannte Hand (angenommen, nur eine Hand wird verarbeitet)
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Bounding Box der Hand berechnen
            h, w, _ = image.shape
            x_min = w
            y_min = h
            x_max = y_max = 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
            
            # Polsterung zur Bounding Box hinzufügen
            pad_x = 20
            pad_y = 20
            x1 = max(x_min - pad_x, 0)
            y1 = max(y_min - pad_y, 0)
            x2 = min(x_max + pad_x, w)
            y2 = min(y_max + pad_y, h)
            
            return (x1, y1, x2, y2)  # Koordinaten der Bounding Box zurückgeben
        
        return None  # Keine Hand erkannt
    
    # Funktion zur Normalisierung der Handposition und -größe
    def normalize_hand(self, image, bbox):
        """Normalisiert die Handposition und -größe"""
        if bbox is None:
            return None  # Nichts zurückgeben, wenn keine Bounding Box vorhanden ist
            
        x_min, y_min, x_max, y_max = bbox
        hand_img = image[y_min:y_max, x_min:x_max]  # Handbereich aus dem Bild ausschneiden
        
        # Auf 128x128 Pixel skalieren
        hand_img = cv2.resize(hand_img, (128, 128))
        
        # In Graustufen konvertieren
        gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
        
        # Adaptive Schwellenwertbildung anwenden
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Rauschen entfernen
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # Morphologisches Öffnen
        
        return thresh  # Schwellenwertbild zurückgeben
    
    # Funktion zur Vorverarbeitung eines einzelnen Bildes
    def preprocess_image(self, image):
        """Vorverarbeite ein einzelnes Bild"""
        # Hand erkennen
        bbox = self.detect_hand(image)  # Bounding Box der Hand erkennen
        if bbox is None:
            return None  # Nichts zurückgeben, wenn keine Hand erkannt wurde
        
        # Hand normalisieren
        processed = self.normalize_hand(image, bbox)  # Hand normalisieren und vorverarbeiten
        if processed is None:
            return None  # Nichts zurückgeben, wenn die Normalisierung fehlschlägt
            
        return processed  # Vorverarbeitetes Bild zurückgeben
    
    # Funktion zur Vorverarbeitung aller Bilder im Datensatz
    def preprocess_dataset(self):
        """Vorverarbeite alle Bilder im Datensatz"""
        print("Starte Datensatz-Vorverarbeitung...")
        
        # Verarbeite jedes Gestenverzeichnis
        for gesture in os.listdir(self.input_dir):
            gesture_dir = os.path.join(self.input_dir, gesture)  # Pfad zum aktuellen Gestenordner
            if not os.path.isdir(gesture_dir):
                continue  # Überspringen, wenn es kein Verzeichnis ist
                
            print(f"\nVerarbeite {gesture}-Gesten...")
            processed_gesture_dir = os.path.join(self.processed_dir, gesture)  # Ausgabeordner für verarbeitete Gesten
            os.makedirs(processed_gesture_dir, exist_ok=True)  # Erstelle den Ordner, falls nicht vorhanden
            
            # Verarbeite jedes Bild
            for img_name in os.listdir(gesture_dir):
                if not img_name.endswith(('.jpg', '.jpeg', '.png')):
                    continue  # Überspringen, wenn es keine Bilddatei ist
                
                img_path = os.path.join(gesture_dir, img_name)  # Vollständiger Pfad zum Bild
                image = cv2.imread(img_path)  # Bild laden
                
                if image is None:
                    print(f"Fehler beim Laden des Bildes: {img_path}")
                    continue  # Überspringen, wenn das Bild nicht geladen werden konnte
                
                processed = self.preprocess_image(image)  # Bild vorverarbeiten
                if processed is not None:
                    # Vorverarbeitetes Bild speichern
                    output_path = os.path.join(processed_gesture_dir, img_name)  # Ausgabepfad
                    cv2.imwrite(output_path, processed)  # Bild speichern
        
        print("\nVorverarbeitung abgeschlossen!")
    
    # Funktion zur Augmentierung des vorverarbeiteten Datensatzes
    def augment_dataset(self):
        """Erweitere den vorverarbeiteten Datensatz mit zusätzlichen Variationen"""
        print("\nStarte Datensatz-Augmentierung...")
        
        for gesture in os.listdir(self.processed_dir):
            gesture_dir = os.path.join(self.processed_dir, gesture)  # Pfad zum verarbeiteten Gestenordner
            if not os.path.isdir(gesture_dir):
                continue  # Überspringen, wenn es kein Verzeichnis ist
                
            print(f"\nAugmentiere {gesture}-Gesten...")
            augmented_gesture_dir = os.path.join(self.augmented_dir, gesture)  # Ausgabeordner für augmentierte Gesten
            os.makedirs(augmented_gesture_dir, exist_ok=True)  # Erstelle den Ordner, falls nicht vorhanden
            
            for img_name in os.listdir(gesture_dir):
                if not img_name.endswith(('.jpg', '.jpeg', '.png')):
                    continue  # Überspringen, wenn es keine Bilddatei ist
                    
                img_path = os.path.join(gesture_dir, img_name)  # Vollständiger Pfad zum Bild
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Bild als Graustufen laden
                
                if image is None:
                    continue  # Überspringen, wenn das Bild nicht geladen werden konnte
                
                # Originalbild speichern
                cv2.imwrite(os.path.join(augmented_gesture_dir, f"orig_{img_name}"), image)
                
                # Rotationsvariationen
                for angle in [-15, 15]:
                    matrix = cv2.getRotationMatrix2D((64, 64), angle, 1.0)  # Rotationsmatrix
                    rotated = cv2.warpAffine(image, matrix, (128, 128))  # Bild rotieren
                    cv2.imwrite(os.path.join(augmented_gesture_dir, f"rot{angle}_{img_name}"), rotated) # Rotiertes Bild speichern
                
                # Helligkeitsvariationen
                for alpha in [0.8, 1.2]:
                    bright = cv2.convertScaleAbs(image, alpha=alpha, beta=0)  # Helligkeit anpassen
                    cv2.imwrite(os.path.join(augmented_gesture_dir, f"bright{alpha}_{img_name}"), bright) # Helligkeitsvariante speichern
                
                # Kontrastvariationen
                for beta in [-30, 30]:
                    contrast = cv2.convertScaleAbs(image, alpha=1.0, beta=beta)  # Kontrast anpassen
                    cv2.imwrite(os.path.join(augmented_gesture_dir, f"contrast{beta}_{img_name}"), contrast) # Kontrastvariante speichern
        
        print("\nAugmentierung abgeschlossen!")

# Beispielnutzung, wenn das Skript direkt ausgeführt wird
if __name__ == "__main__":
    preprocessor = ImagePreprocessor()  # Erstelle eine Instanz des Vorverarbeiters
    preprocessor.preprocess_dataset()  # Führe die Dataset-Vorverarbeitung aus