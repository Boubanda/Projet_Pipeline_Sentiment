# Étape 1: Utiliser une image Python de base
FROM python:3.11-slim

# Étape 2: Définir le répertoire de travail
WORKDIR /app

# Étape 3: Copier tous les fichiers nécessaires du projet dans l'image Docker
COPY . /app

# Étape 4: Installer les dépendances depuis le fichier requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Ajouter Gunicorn si ce n'est pas dans requirements.txt
RUN pip install gunicorn

# Étape 5: Exposer le port utilisé par l'application (5000 pour Flask)
EXPOSE 5000

# Étape 6: Commande pour démarrer le serveur Flask avec Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
