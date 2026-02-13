import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def run_analysis():
    print("--- CARICAMENTO DATASET ---")
    try:
        df = pd.read_csv("connect4_dataset_hq.csv")
    except FileNotFoundError:
        print("ERRORE: Esegui prima 'generate_dataset.py'!")
        return

    X = df.iloc[:, 0:42].values
    y = df.iloc[:, 42].values

    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training su {len(X_train)} campioni, Test su {len(X_test)} campioni.")

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), 
                        activation='relu', 
                        max_iter=500, 
                        random_state=42,
                        early_stopping=True, # Importante per vedere la validation loss
                        validation_fraction=0.1)

    print("--- ADDESTRAMENTO IN CORSO... ---")
    mlp.fit(X_train, y_train)

    plt.figure(figsize=(10, 6))
    plt.plot(mlp.loss_curve_, label='Training Loss', color='blue')
    plt.title('Curva di Convergenza (Loss) - MLP', fontsize=14)
    plt.xlabel('Iterazioni (Epoche)', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('images/training_loss_curve.png') # Assicurati che la cartella images esista
    print("Grafico Loss salvato in 'images/training_loss_curve.png'")

    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n--- RISULTATI TEST SET ---")
    print(f"Accuratezza: {acc:.2%}")
    print("\nReport di Classificazione:")
    print(classification_report(y_test, y_pred, target_names=['AI Wins (-1)', 'Draw (0)', 'Player Wins (1)']))

    cm = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['AI', 'Draw', 'Player'])
    
    plt.figure(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Matrice di Confusione')
    plt.savefig('images/confusion_matrix.png')
    print("Matrice Confusione salvata in 'images/confusion_matrix.png'")

if __name__ == "__main__":
    import os
    if not os.path.exists('images'):
        os.makedirs('images')
    run_analysis()