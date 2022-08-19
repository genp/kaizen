# Start building this docker from a known image
FROM python:3.9.6 # or from pytorch wrapper


WORKDIR /python-docker
# need pg_config: sudo apt-get install libpq-dev python-dev
# install pip : sudo apt install python3-pip
# intall flask? this doesn't seem to happen with only the pip on the ubuntu ami: sudo apt install python3-flask
#install celery: sudo apt install python-celery-common
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

# This will only support http not https
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]