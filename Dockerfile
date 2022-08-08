# TODO: build this docker from which existing image
FROM python:3.9

# TODO: system stuff, zsh, git
RUN apk --update add zsh

# -- to install python package psycopg2 (for postgres) -- #
# RUN apt-get update
# RUN apt-get install -y postgresql libpq-dev postgresql-client postgresql-client-common gcc

# add user (change to whatever you want)
# prevents running sudo commands
RUN useradd -r -s /bin/bash app-user

# set current env
ENV HOME /app
WORKDIR /app
ENV PATH="/app/.local/bin:${PATH}"

RUN chown -R app-user:app-user /app
USER app-user

# set app config option
ENV FLASK_ENV=production

# set argument vars in docker-run command
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION
# -- AWS RDS vars -- #
# ARG POSTGRES_USER
# ARG POSTGRES_PW
# ARG POSTGRES_URL
# ARG POSTGRES_DB

ENV AWS_ACCESS_KEY_ID $AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY $AWS_SECRET_ACCESS_KEY
ENV AWS_DEFAULT_REGION $AWS_DEFAULT_REGION

# Avoid cache purge by adding requirements first
ADD ./requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r ./requirements.txt --user

# Add the rest of the files
COPY . /app
WORKDIR /app

# TODO: setup database defaults, forge data

# start web server
CMD ["flask", "run"]

# TODO: start celery server
# celery -A tasks.celery worker --loglevel=debug