import cv2
import numpy as np
import torch
from models.model import GestureClassifier
from preprocessing.preprocess_images import ImagePreprocessor

# Hauptfunktion des Testskripts
def main():
    # Initialisiere das Modell
    print("Initialisiere GestureClassifier...")
    model = GestureClassifier()  # Erstelle eine Instanz des Gestenklassifizierers
    model.load_model('Gesture Control/models/saved/best_model')  # Lade das trainierte Modell
    model.eval()  # Setze das Modell in den Evaluationsmodus
    
    # Initialisiere den Vorverarbeiter
    print("Initialisiere ImagePreprocessor...")
    preprocessor = ImagePreprocessor()  # Erstelle eine Instanz des Bildvorverarbeiters
    
    # Kamera initialisieren
    print("Versuche Kamera zu öffnen...")
    cap = cv2.VideoCapture(0)  # Video-Capture-Objekt initialisieren (Webcam)
    # Überprüfen, ob die Kamera geöffnet werden konnte
    if not cap.isOpened():
        print("Fehler: Kamera konnte nicht geöffnet werden. Bitte überprüfen Sie die Kamerakonnektivität und Berechtigungen.")
        return  # Programm beenden, wenn Kamera nicht verfügbar
    print("Kamera erfolgreich geöffnet.")
    
    # Gesten-Labels und zugehörige Aktionen
    gesture_labels = ['thumb', 'palm', 'fist']  # Liste der erkannten Gesten
    gesture_actions = {
        'thumb': 'Start Musik',
        'palm': 'Pause Musik',
        'fist': 'Stop Musik'
    }
    
    print("\nGestenerkennung gestartet!")
    print("Drücken Sie 'q' zum Beenden")
    
    while True:  # Endlosschleife für die Echtzeit-Erkennung
        ret, frame = cap.read()  # Frame von der Kamera lesen
        if not ret:
            print("Fehler: Konnte keinen Frame von der Kamera lesen. Beende.")
            break  # Schleife beenden, wenn kein Frame gelesen werden konnte
        
        # Frame horizontal spiegeln für eine spätere Selfie-Ansicht
        frame = cv2.flip(frame, 1)
        
        # Frame vorverarbeiten
        processed = preprocessor.preprocess_image(frame)  # Bild vorverarbeiten
        
        gesture_name = None  # Name der erkannten Geste
        action = None  # Zugehörige Aktion
        confidence = 0.0  # Konfidenz der Erkennung
        
        if processed is not None:  # Wenn ein vorverarbeitetes Bild vorhanden ist
            # Normalisiere Pixelwerte auf den Bereich [0, 1]
            processed = processed.astype(np.float32) / 255.0
            # Glätten für FFNN (Feed Forward Neural Network)
            processed_flat = processed.flatten().reshape(1, -1)  # Bild glätten und Form anpassen
            with torch.no_grad():  # Deaktiviere die Gradientenberechnung
                outputs = model(torch.FloatTensor(processed_flat))  # Vorhersage mit dem Modell
                probabilities = torch.softmax(outputs, dim=1)[0]  # Wahrscheinlichkeiten der Klassen
                predicted_class = torch.argmax(probabilities).item()  # Klasse mit der höchsten Wahrscheinlichkeit
                confidence = probabilities[predicted_class].item()  # Konfidenzwert
                gesture_name = gesture_labels[predicted_class]  # Gestenname
                action = gesture_actions[gesture_name]  # Zugehörige Aktion
        
        # Geste und Aktion anzeigen
        if gesture_name is not None:
            cv2.putText(frame, f"Geste: {gesture_name}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)  # Geste anzeigen
            cv2.putText(frame, f"Aktion: {action}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)  # Aktion anzeigen
            cv2.putText(frame, f"Konfidenz: {confidence:.2f}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)  # Konfidenz anzeigen
        else:
            cv2.putText(frame, "Zeigen Sie Ihre Handgeste", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)  # Hinweis anzeigen
        
        # Frame anzeigen
        cv2.imshow('Gestenerkennung', frame)  # Frame im Fenster anzeigen
        print("cv2.imshow aufgerufen.")
        
        # Tastendrücke verarbeiten
        key = cv2.waitKey(30) & 0xFF  # Auf Tastendruck warten
        if key == ord('q'):  # Wenn 'q' gedrückt wird
            break  # Schleife beenden
    
    # Aufräumen
    print("Kamera wird freigegeben und Fenster werden zerstört...")
    cap.release()  # Kamera freigeben
    cv2.destroyAllWindows()  # Alle OpenCV-Fenster schließen

# Überprüfen, ob das Skript direkt ausgeführt wird
if __name__ == "__main__":
    main()  # Rufe die Hauptfunktion auf, um das Programm zu starten