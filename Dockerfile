FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e ".[dev,analysis]" jupyterlab

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
