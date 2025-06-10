import cv2
import numpy as np
from skimage.feature import hog
from skimage import measure

# Definition der Klasse FeatureExtractor
class FeatureExtractor:
    # Konstruktor der Klasse
    def __init__(self):
        self.hog_orientations = 9  # Anzahl der Orientierungs-Bins für HOG
        self.hog_pixels_per_cell = (8, 8)  # Größe der Zelle in Pixeln für HOG
        self.hog_cells_per_block = (2, 2)  # Anzahl der Zellen pro Block für HOG
    
    # Funktion zum Extrahieren von HOG-Merkmalen aus einem Bild
    def extract_hog_features(self, image):
        """Extrahiert HOG-Merkmale aus dem Bild."""
        features = hog(
            image,
            orientations=self.hog_orientations,
            pixels_per_cell=self.hog_pixels_per_cell,
            cells_per_block=self.hog_cells_per_block,
            visualize=False  # Keine Visualisierung der HOG-Merkmale
        )
        return features  # HOG-Merkmale zurückgeben
    
    # Funktion zum Extrahieren von formbasierten Merkmalen aus einem Bild
    def extract_shape_features(self, image):
        """Extrahiert formbasierte Merkmale aus dem Bild."""
        # In Binärbild konvertieren (Schwellenwert 0.5)
        binary = (image > 0.5).astype(np.uint8)
        
        # Konturen finden
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:  # Wenn keine Konturen gefunden wurden
            return np.zeros(7)  # Array mit Nullen zurückgeben
        
        # Größte Kontur erhalten
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Formmerkmale berechnen
        area = cv2.contourArea(largest_contour)  # Fläche der Kontur
        perimeter = cv2.arcLength(largest_contour, True)  # Umfang der Kontur
        
        # Konvexe Hülle und deren Fläche berechnen
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        
        # Solidität berechnen (Verhältnis von Konturfläche zu Hüllenfläche)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # Seitenverhältnis berechnen
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Hu-Momente berechnen (Invarianten zur Formbeschreibung)
        moments = cv2.moments(largest_contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Alle Formmerkmale kombinieren
        shape_features = np.array([
            area,
            perimeter,
            solidity,
            aspect_ratio,
            hu_moments[0],
            hu_moments[1],
            hu_moments[2]
        ])
        
        return shape_features  # Formmerkmale zurückgeben
    
    # Funktion zum Extrahieren von statistischen Merkmalen aus einem Bild
    def extract_statistical_features(self, image):
        """Extrahiert statistische Merkmale aus dem Bild."""
        # Bild in ein 4x4-Gitter unterteilen
        h, w = image.shape
        cell_h, cell_w = h // 4, w // 4
        
        features = []
        for i in range(4):
            for j in range(4):
                cell = image[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]  # Zelle auswählen
                features.extend([
                    np.mean(cell),  # Mittelwert der Zelle
                    np.std(cell),  # Standardabweichung der Zelle
                    np.max(cell),  # Maximalwert der Zelle
                    np.min(cell)  # Minimalwert der Zelle
                ])
        
        return np.array(features)  # Statistische Merkmale zurückgeben
    
    # Funktion zum Extrahieren aller Merkmale aus einem Bild
    def extract_all_features(self, image):
        """Extrahiert alle Merkmale aus dem Bild."""
        hog_features = self.extract_hog_features(image)  # HOG-Merkmale extrahieren
        shape_features = self.extract_shape_features(image)  # Formmerkmale extrahieren
        statistical_features = self.extract_statistical_features(image)  # Statistische Merkmale extrahieren
        
        # Alle Merkmale kombinieren (konkatenieren)
        all_features = np.concatenate([
            hog_features,
            shape_features,
            statistical_features
        ])
        
        return all_features  # Alle Merkmale zurückgeben
    
    # Funktion zur Normalisierung von Merkmalen
    def normalize_features(self, features):
        """Normalisiert Merkmale auf null Mittelwert und Einheitsvarianz."""
        mean = np.mean(features, axis=0)  # Mittelwert pro Merkmal
        std = np.std(features, axis=0)  # Standardabweichung pro Merkmal
        std[std == 0] = 1  # Division durch Null vermeiden
        normalized = (features - mean) / std  # Normalisierung
        return normalized  # Normalisierte Merkmale zurückgeben

# Beispielnutzung, wenn das Skript direkt ausgeführt wird
if __name__ == "__main__":
    # Beispielverwendung
    extractor = FeatureExtractor()  # Erstelle eine Instanz des FeatureExtractors
    
    # Lade ein Beispielbild
    image = cv2.imread('data/processed/thumb/sample.jpg', cv2.IMREAD_GRAYSCALE)  # Bild laden
    if image is not None:
        # Bild normalisieren
        image = image / 255.0
        
        # Merkmale extrahieren
        features = extractor.extract_all_features(image)  # Alle Merkmale extrahieren
        print(f"Gesamtzahl der Merkmale: {len(features)}")  # Anzahl der Merkmale ausgeben
        print(f"Merkmalvektor: {features}")  # Merkmalvektor ausgeben 