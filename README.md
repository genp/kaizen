## Webserver for Crowd-in-the-Loop Active learning.

For further info on the use of this system, please see the following papers:


[Tropel: Crowdsourcing Detectors with Minimal Training]: http://cs.brown.edu/~gmpatter/pub_papers/patterson_hcomp2015.pdf
[Kaizen: The Crowd Pathologist]: http://cs.brown.edu/people/gmpatter/groupsight/kaizen.pdf

## Contents of this Repo

LICENSE - Copyright genp
README - This file
app - The main flask webserver
bin - scripts for setting up db, recreating, and running main server
 db_create.py
 kaizen.py
 db_migrate.py
 reset-all

## Prerequiste for running Kaizen
Linux:
 sudo yum install blas-devel lapack-devel gcc-c++ freetype-devel libpng-devel libffi-devel libopenssl-devel postgresql94-devel
 sudo chkconfig postgresql94 on
? yum install postgresql94-server
 sudo service postgresql96 initdb
 sudo service postgresql96 start

? sudo pip install virtualenvwrapper
? source /usr/local/bin/virtualenvwrapper.sh
? sudo yum install openssl-devel cmake

## To Begin: 

git clone git@github.com:genp/kaizen.git
mkvirtualenv kaizen
cd kaizen
setvirtualenvproject kaizen
ln -sf `pwd`/bin/postactivate ~/.virtualenvs/kaizen/bin
ln -sf `pwd`/bin/postdeactivate ~/.virtualenvs/kaizen/bin
pip install --upgrade pip
if ! python -c "import numpy" > /dev/null 2>1 ; then pip install numpy==1.9.2; fi
pip install -r pip-freeze.txt
add2virtualenv .
echo "backend      : Agg" > /home/$USER/matplotlibrc

### If not already installed
gem install sass

### Install opencv:
  OS X:
   brew install homebrew/science/opencv
   ln -s /usr/local/Cellar/opencv/2*/lib/python2.7/site-packages/cv.py \
     ~/.virtualenvs/kaizen/lib/python*/site-packages
   ln -s /usr/local/Cellar/opencv/2*/lib/python2.7/site-packages/cv2.so \
     ~/.virtualenvs/kaizen/lib/python*/site-packages

  Linux: 
    git clone https://github.com/Itseez/opencv.git
    cd ~/opencv
    mkdir release
    cd release
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
    make
    sudo make install

### Install opencv3:
  OS X:
   brew install opencv
   ln -s /usr/local/Cellar/opencv3/3*/lib/python2.7/site-packages/cv2.so \
     ~/.virtualenvs/kaizen/lib/python*/site-packages
  Linux: ???

### For OS X:
  <INSTALL caffe>
  pip install protobuf

### To suppress pycaffe terminal output:
  os.environ['GLOG_minloglevel'] = '2' 

### Install Caffe:
    git clone https://github.com/BVLC/caffe.git
    cp Makefile.config.example Makefile.config
    #Edit config for proper system settings
    make all
    make test
    make runtest
    pip install -r python/requirements.txt
    make pycaffe
    add2virtualenv ~/caffe
        

#### Install dependencies first:
    brew install protobuf
    brew install boost boost-python
    brew install gflags
    brew install glog
    wget https://raw.github.com/Homebrew/homebrew-science/master/hdf5.rb
    mv hdf5.rb /usr/local/Library/Formula
    brew install hdf5
    brew install leveldb
    brew install lmdb
    brew install homebrew/science/opencv
    edit caffe/Makefile.config to install CPU only version

### Create Database and start Kaizen and Celery Servers:
### ----------------------------------------------------
sudo -u postgres createuser -sdr $USER
#change ident to trust in pg_hba.conf
createdb kaizen
./bin/db_create.py
./bin/kaizen.py
celery -A tasks.celery worker --loglevel=debug


