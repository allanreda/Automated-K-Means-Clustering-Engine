# docker build -t clustering_app .
# docker run --rm -e PORT=3000 -p 8080:8080 --env-file=.env clustering_app

FROM python:3.12-alpine

# Install the required system packages for building Python packages and lscpu utility
RUN apk update && apk add --no-cache \
    gcc \
    g++ \
    musl-dev \
    util-linux  # Install lscpu system utility for parallel processing in scikit-learn

WORKDIR /src

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip python -m pip install -r requirements.txt

COPY . .

EXPOSE 8080

ENTRYPOINT ["gunicorn", "-b", ":8080", "app:app", "--timeout", "300", "--workers", "5"]