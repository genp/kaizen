## Webserver for Human-in-the-Loop Active learning.
![The Kaizen Query Response UI](https://raw.githubusercontent.com/genp/kaizen/main/app/static/images/query_page.png)

For further info on the use of this system, please see the following papers:

[Tropel: Crowdsourcing Detectors with Minimal Training](http://cs.brown.edu/~gmpatter/pub_papers/patterson_hcomp2015.pdf)

[Kaizen: The Crowd Pathologist](http://cs.brown.edu/people/gmpatter/groupsight/kaizen.pdf)

## Contents of this Repo

    LICENSE - Open source
    README - This file
    app - The main flask webserver
    bin - scripts for setting up db, recreating, and running main server
    test - unit tests
    utils - utility functions for computer vision and feature extraction
    tasks.py - jobs for the Celery task queue
    extract.py - functions for extracting features
    requirements.txt - python requirements
    Dockerfile - instructions for Dockerizing this repo

## Set up venv:
> mkvirtualenv kaizen # this app tested with Python3.9
> pip install -r requirements.txt

To restart venv:
> workon kaizen

To add to python requirements:
> pip freeze > requirements.txt

Setup the configuration parameters:
>cp config-example.py config.py
... Do some editing ...

Add this repo to your PYTHONPATH:
> export PYTHONPATH=/Users/$USER/kaizen:$PYTHONPATH

## Setup the postgres database:
> createdb <APPNAME>-local # APPNAME should match that set in config.py

If you'd like to blow away the db:
> dropdb <APPNAME>-local

Load some data to play with:
> ./bin/forge.py

## Start Task Queue
Start the Celery server to manage new tasks from the Flask app:
>  celery -A tasks.celery worker --loglevel=debug

## Start App Server
Run Server:
> flask run

## Unit Testing
Tests are in the tests directory
> cd tests
Run unit test classes using:
> python -m unittest unit_tests.py

The forge functions also serve as system tests.

## License

This project is licensed under the MIT License (see the
[LICENSE](LICENSE) file for details).
