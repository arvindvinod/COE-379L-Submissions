FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    flask==2.3.2 \
    tensorflow==2.15.0 \
    pillow==10.0.0 \
    numpy==1.24.3 \
    gunicorn==21.2.0

COPY app.py final_best_model.h5 ./

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]