import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Definition der Handgesten-Datensatzklasse
class HandGestureDataset(Dataset):
    def __init__(self, X, y):
        # Glätte die Bilder für den FFNN-Input (Feed Forward Neural Network)
        self.X = torch.FloatTensor(X.reshape(X.shape[0], -1))
        self.y = torch.LongTensor(y)
    
    # Gibt die Anzahl der Samples im Datensatz zurück
    def __len__(self):
        return len(self.X)
    
    # Gibt ein Sample (Bild und Label) basierend auf dem Index zurück
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Definition der Gestenklassifizierer-Modellklasse
class GestureClassifier(nn.Module):
    # Konstruktor des Modells
    def __init__(self, input_size=128*128, num_classes=3):
        super(GestureClassifier, self).__init__()
        
        # Feed Forward Neuronales Netzwerk (FFNN)
        self.model = nn.Sequential(
            # Erste versteckte Schicht
            nn.Linear(input_size, 1024),  # Lineare Transformation
            nn.ReLU(),  # Aktivierungsfunktion (Rectified Linear Unit)
            nn.BatchNorm1d(1024),  # Batch-Normalisierung
            nn.Dropout(0.3),  # Dropout zur Regularisierung (30% der Neuronen werden deaktiviert)
            
            # Zweite versteckte Schicht
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            # Dritte versteckte Schicht
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            # Ausgabeschicht
            nn.Linear(256, num_classes)  # Ausgabe in die Anzahl der Klassen
        )
        
        # Verschiebe das Modell auf die GPU, falls verfügbar, sonst CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        print(f"Verwende Gerät: {self.device}")
    
    # Forward-Pass des Modells
    def forward(self, x):
        return self.model(x)
    
    # Trainingsfunktion des Modells
    def train_model(self, X, y, batch_size=32, epochs=50, learning_rate=0.001, patience=10):
        # Daten aufteilen in Trainings- und Validierungssets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Erstelle Datensätze und Dataloader
        train_dataset = HandGestureDataset(X_train, y_train)
        val_dataset = HandGestureDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Verlustfunktion und Optimierer
        criterion = nn.CrossEntropyLoss()  # Kreuzentropie-Verlustfunktion
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)  # Adam-Optimierer
        # Lernraten-Scheduler: Reduziert die Lernrate, wenn sich die Validierungsgenauigkeit nicht verbessert
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        # Trainings-Loop
        best_val_acc = 0  # Beste Validierungsgenauigkeit
        epochs_no_improve = 0  # Zähler für Epochen ohne Verbesserung (für Early Stopping)
        for epoch in range(epochs):
            self.train()  # Setze das Modell in den Trainingsmodus
            train_loss = 0  # Trainingsverlust für die aktuelle Epoche
            train_correct = 0  # Anzahl der korrekt klassifizierten Trainings-Samples
            train_total = 0  # Gesamtzahl der Trainings-Samples
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)  # Daten auf das Gerät verschieben
                
                optimizer.zero_grad()  # Gradienten zurücksetzen
                outputs = self(batch_X)  # Forward-Pass
                loss = criterion(outputs, batch_y)  # Verlust berechnen
                loss.backward()  # Backpropagation (Gradienten berechnen)
                optimizer.step()  # Modellparameter aktualisieren
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)  # Vorhergesagte Klassen erhalten
                train_total += batch_y.size(0)  # Gesamtzahl der Samples im Batch
                train_correct += (predicted == batch_y).sum().item()  # Korrekt klassifizierte Samples
            
            # Validierung
            self.eval()  # Setze das Modell in den Evaluationsmodus (deaktiviert Dropout etc.)
            val_correct = 0  # Anzahl der korrekt klassifizierten Validierungs-Samples
            val_total = 0  # Gesamtzahl der Validierungs-Samples
            all_preds = []  # Liste zur Speicherung aller Vorhersagen
            all_labels = []  # Liste zur Speicherung aller echten Labels
            
            with torch.no_grad():  # Deaktiviere die Gradientenberechnung für die Validierung
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
                    all_preds.extend(predicted.cpu().numpy())  # Vorhersagen speichern
                    all_labels.extend(batch_y.cpu().numpy())  # Labels speichern
            
            train_acc = 100 * train_correct / train_total  # Trainingsgenauigkeit berechnen
            val_acc = 100 * val_correct / val_total  # Validierungsgenauigkeit berechnen
            
            # Lernrate aktualisieren basierend auf der Validierungsgenauigkeit
            scheduler.step(val_acc)
            
            print(f'Epoche [{epoch+1}/{epochs}]')
            print(f'Trainingsverlust: {train_loss/len(train_loader):.4f}, Trainingsgenauigkeit: {train_acc:.2f}%')
            print(f'Validierungsgenauigkeit: {val_acc:.2f}%')
            
            # Bestes Modell speichern und auf Early Stopping prüfen
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('Gesture Control/models/saved/best_model')  # Modell speichern
                epochs_no_improve = 0  # Zähler zurücksetzen, wenn sich die Validierungsgenauigkeit verbessert
            else:
                epochs_no_improve += 1  # Zähler erhöhen, wenn keine Verbesserung
                if epochs_no_improve == patience:
                    print(f'Early Stopping ausgelöst nach {patience} Epochen ohne Verbesserung der Validierungsgenauigkeit.')
                    break  # Training beenden
        
        # Endgültige Ergebnisse ausgeben
        print("\nKlassifikationsbericht:")
        print(classification_report(all_labels, all_preds, target_names=['thumb', 'palm', 'fist']))
        print("\nKonfusionsmatrix:")
        print(confusion_matrix(all_labels, all_preds))
    
    # Vorhersagefunktion des Modells
    def predict(self, X):
        self.eval()  # Setze das Modell in den Evaluationsmodus
        with torch.no_grad():  # Deaktiviere die Gradientenberechnung
            # Eingabe glätten, falls erforderlich
            if len(X.shape) > 2:
                X = X.reshape(X.shape[0], -1)
            
            X = torch.FloatTensor(X).to(self.device)  # Daten auf das Gerät verschieben
            outputs = self(X)  # Forward-Pass
            _, predicted = torch.max(outputs.data, 1)  # Vorhergesagte Klasse erhalten
            return predicted.cpu().numpy()  # Vorhersagen als NumPy-Array zurückgeben
    
    # Funktion zum Speichern des Modells
    def save_model(self, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Erstelle Verzeichnisse, falls nicht vorhanden
        torch.save({
            'model_state_dict': self.state_dict(),  # Speichere die Modellgewichte
            'input_size': self.model[0].in_features  # Speichere die Eingabegröße
        }, f"{model_path}.pt")  # Speichere das Modell im PyTorch-Format (.pt)
    
    # Funktion zum Laden des Modells
    def load_model(self, model_path):
        checkpoint = torch.load(f"{model_path}.pt", map_location=self.device)  # Lade das Modell-Checkpoint
        # Setze die Eingabegröße der ersten Schicht zurück, falls sie sich geändert hat
        self.model[0] = nn.Linear(checkpoint['input_size'], 1024)
        self.load_state_dict(checkpoint['model_state_dict'])  # Lade die gespeicherten Modellgewichte
        self.to(self.device)  # Modell auf das Gerät verschieben

# Beispielnutzung, wenn das Skript direkt ausgeführt wird
if __name__ == "__main__":
    # Beispielverwendung
    # Laden und Vorbereiten Ihrer Daten hier
    # X = ...  # Ihre Feature-Matrix
    # y = ...  # Ihre Labels
    
    # Erstellen und Trainieren verschiedener Modelle (auskommentiert)
    models = ['svm', 'random_forest', 'knn', 'decision_tree']
    for model_type in models:
        print(f"\nTrainiere {model_type}-Modell...")
        # classifier = GestureClassifier(model_type=model_type)
        # classifier.train(X, y) 