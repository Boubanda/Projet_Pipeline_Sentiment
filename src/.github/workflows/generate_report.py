import json
import time

def generate_report(prediction, text):
    report = {
        "timestamp": time.time(),
        "text": text,
        "prediction": prediction
    }
    
    with open('performance_report.json', 'a') as f:
        json.dump(report, f)
        f.write("\n")

if __name__ == "__main__":
    text = "I love this product!"
    prediction = 1  # Remplacez par votre modèle de prédiction
    generate_report(prediction, text)
