# Verwende das offizielle Miniconda-Image als Basisimage
FROM continuumio/miniconda3

# Setze das Arbeitsverzeichnis im Container
WORKDIR /app

# Kopiere die Anforderungen (dependencies) in den Container
COPY app/requirements.txt app/

#VOLUME /app/data

# Kopiere die anderen Dateien
COPY app/app.py app/
COPY app/forecast.py app/
#COPY data /app/data

# Erstelle und aktiviere eine neue Conda-Umgebung und installiere die Abhängigkeiten
RUN conda create -y --name myenv python=3.10 && \
    conda run -n myenv pip install --upgrade pip && \
    conda run -n myenv pip install -r app/requirements.txt && \
    conda clean -afy

# Stelle sicher, dass die Conda-Umgebung beim Containerstart aktiv ist
ENV PATH="/opt/conda/envs/myenv/bin:$PATH"

# Exponiere den Port, auf dem die Flask-Anwendung läuft
EXPOSE 5000


# Starte die Flask-Anwendung beim Ausführen des Containers
CMD ["/bin/bash", "-c", "source activate myenv && python app/app.py"]