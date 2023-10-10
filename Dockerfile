FROM python:3.10.13-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock" , "./"] /app/

RUN pipenv install --system --deploy

COPY ["predict.py", "model_C=10.bin" , "app.py", "./"] /app/

ENTRYPOINT ["gunicorn","--bind=0.0.0.0:9000","app:app"]

EXPOSE 9000
