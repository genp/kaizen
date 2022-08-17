# Start building this docker from a known image
FROM python:3.9.6 # or from pytorch wrapper


WORKDIR /python-docker
# need pg_config: sudo apt-get install libpq-dev python-dev
# install pip : sudo apt install python3-pip
# isntall flask? this doesn't seem to happen with only the pip on ubuntu..." sudo apt install python3-flask
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]