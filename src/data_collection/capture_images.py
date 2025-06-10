import cv2
import os
import mediapipe as mp
import numpy as np

# Funktion zur Überprüfung, ob eine Geste gut positioniert ist
def is_good_gesture(hand_landmarks, frame_shape):
    """Check if the hand gesture is well-positioned and clear."""
    h, w = frame_shape[:2] 
    
    # Get hand center
    center_x = sum(lm.x * w for lm in hand_landmarks.landmark) / len(hand_landmarks.landmark)
    center_y = sum(lm.y * h for lm in hand_landmarks.landmark) / len(hand_landmarks.landmark)
    
    # Check if hand is centered in frame (within 40% of center)
    if not (0.3 * w <= center_x <= 0.7 * w and 0.3 * h <= center_y <= 0.7 * h):
        return False
    
    # Check if hand is not too small or too large
    x_min = min(lm.x * w for lm in hand_landmarks.landmark)
    x_max = max(lm.x * w for lm in hand_landmarks.landmark)
    y_min = min(lm.y * h for lm in hand_landmarks.landmark)
    y_max = max(lm.y * h for lm in hand_landmarks.landmark)
    
    hand_width = x_max - x_min
    hand_height = y_max - y_min
    
    # Hand should be between 20% and 60% of frame width
    if not (0.2 * w <= hand_width <= 0.6 * w):
        return False
    
    return True

# Hauptfunktion zum Erfassen von Bildern
def main():
    gestures = ['thumb', 'palm', 'fist']  # Definierte Gesten
    print("Verfügbare Gesten:", gestures)
    gesture = input("Gestenname eingeben: ").strip().lower()  # Benutzer zur Eingabe einer Geste auffordern
    # Überprüfen, ob die eingegebene Geste gültig ist
    if gesture not in gestures:
        print(f"Ungültige Geste. Bitte wählen Sie aus: {gestures}")
        return  # Programm beenden, wenn die Geste ungültig ist

    save_dir = f"Gesture Control/data/raw/{gesture}"  # Verzeichnis zum Speichern der Bilder
    os.makedirs(save_dir, exist_ok=True)  # Erstelle das Verzeichnis, falls es nicht existiert
    cap = cv2.VideoCapture(0)  # Video-Capture-Objekt initialisieren (Webcam)
    img_count = 0  # Zähler für aufgenommene Bilder

    mp_hands = mp.solutions.hands  # MediaPipe Hands Modul
    hands = mp_hands.Hands(  # Konfiguration des MediaPipe Hands Moduls
        static_image_mode=False,  # Für Video-Streams, nicht statische Bilder
        max_num_hands=1,  # Maximale Anzahl der zu erkennenden Hände
        min_detection_confidence=0.7,  # Minimale Erkennungssicherheit
        min_tracking_confidence=0.7  # Minimale Tracking-Sicherheit
    )
    mp_draw = mp.solutions.drawing_utils  # Dienstprogramm zum Zeichnen von Landmarken

    print("\nAnweisungen zur Aufnahme:")
    print("1. Halten Sie Ihre Hand mittig im Rahmen")
    print("2. Halten Sie einen konstanten Abstand zur Kamera ein")
    print("3. Sorgen Sie für gute Beleuchtung")
    print("4. Drücken Sie 'c' zum Aufnehmen, wenn die Hand gut positioniert ist")
    print("5. Drücken Sie 'q' zum Beenden")
    print("\nAufnahme wird gestartet...")

    while True:  # Endlosschleife zur Videoaufnahme
        ret, frame = cap.read()  # Frame von der Kamera lesen
        if not ret:
            break  # Schleife beenden, wenn kein Frame gelesen werden konnte
        frame = cv2.flip(frame, 1)  # Frame horizontal spiegeln (Selfie-Ansicht)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Frame von BGR zu RGB konvertieren
        results = hands.process(rgb)  # Hand-Landmarken mit MediaPipe verarbeiten
        hand_detected = False  # Flag: Hand erkannt
        good_gesture = False  # Flag: Geste gut positioniert
        
        if results.multi_hand_landmarks:  # Wenn Hand-Landmarken erkannt wurden
            for hand_landmarks in results.multi_hand_landmarks:
                # Bounding Box der Hand erhalten
                h, w, _ = frame.shape
                x_min = w
                y_min = h
                x_max = y_max = 0
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                
                # Gleichmäßige Polsterung hinzufügen
                pad_x = 100  # Reduzierte Polsterung für konsistentere Aufnahmen
                pad_y = 100  # Gleichmäßige Polsterung oben und unten
                
                # Box-Koordinaten mit Polsterung berechnen
                x1 = max(x_min - pad_x, 0)
                y1 = max(y_min - pad_y, 0)
                x2 = min(x_max + pad_x, w)
                y2 = min(y_max + pad_y, h)
                
                # Überprüfen, ob die Geste gut positioniert ist
                good_gesture = is_good_gesture(hand_landmarks, frame.shape)
                
                # Rechteck mit Farbe basierend auf der Qualität zeichnen
                color = (0, 255, 0) if good_gesture else (0, 0, 255)  # Grün für gut, Rot für schlecht
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Rechteck zeichnen
                
                # Hand-Landmarken zeichnen
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                hand_detected = True  # Hand erkannt
                roi = frame[y1:y2, x1:x2]  # Region of Interest (Handbereich)
                break

        # Anweisungen und Status hinzufügen
        if hand_detected:
            if good_gesture:
                msg = "Gute Position! Drücken Sie 'c' zum Aufnehmen."
                color = (0, 255, 0)
            else:
                msg = "Handposition anpassen (mittig im Rahmen)"
                color = (0, 0, 255)
        else:
            msg = "Zeigen Sie Ihre Hand der Kamera"
            color = (0, 0, 255)
            
        cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)  # Nachricht anzeigen
        cv2.putText(frame, f"Aufgenommene Bilder: {img_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2) # Bildzähler anzeigen
        cv2.putText(frame, f"Geste: {gesture}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2) # Aktuelle Geste anzeigen
        
        cv2.imshow("Handgesten-Aufnahme", frame)  # Frame anzeigen
        key = cv2.waitKey(1) & 0xFF  # Tastendruck abfragen
        # Bild aufnehmen, wenn 'c' gedrückt wird, Hand erkannt und Geste gut positioniert ist
        if key == ord('c') and hand_detected and good_gesture:
            img_path = os.path.join(save_dir, f"{gesture}_{img_count}.jpg")  # Dateipfad für das Bild
            cv2.imwrite(img_path, roi)  # Bild speichern
            img_count += 1  # Bildzähler erhöhen
            print(f"Bild {img_count} aufgenommen! Bewegen Sie Ihre Hand in eine neue Position.")
        elif key == ord('q'):  # Schleife beenden, wenn 'q' gedrückt wird
            break
    cap.release()  # Kamera freigeben
    cv2.destroyAllWindows()  # Alle OpenCV-Fenster schließen

# Überprüfen, ob das Skript direkt ausgeführt wird
if __name__ == "__main__":
    main()  # Rufe die Hauptfunktion auf, um das Programm zu starten 