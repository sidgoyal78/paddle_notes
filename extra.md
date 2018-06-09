### Development for book repository


Following are the steps for localhost:

- Pull the latest 

	docker pull paddlepaddle/paddle:latest

- Install go using https://golang.org/doc/install
  (reason: for converting stuff from md to ipynb)

- Export gopath, by first creating a temp directory:

       mkdir godir

       export GOPATH=${PWD}

       echo ${GOPATH} => /home/sidgoyal/godir


- git clone <your fork of book>
- git clone  https://github.com/sidgoyal78/book.git

- cd book/.tools
- Run the conversion: ./convert-markdown-into-ipynb-and-test.sh 
(This prepares ipynb files in each of the directories, you can see a .ipynb file in "book/01.fit_a_line")

Running latest docker image: 
- docker run -it -p 127.0.0.1:8088:8888 -v $PWD:/work paddlepaddle/paddle:latest /bin/bash

- install Jupyter notebook inside the container using "pip install jupyter"

- Run jupyter:
  jupyter notebook --allow-root --ip=0.0.0.0 --no-browser --port 8888

- Copy the URL 

http://86fc36e2f382:8888/?token=e093f12b2fea5d953ace2756b98ca63e8cbccf1fbc6673bc&token=e093f12b2fea5d953ace2756b98ca63e8cbccf1fbc6673bc

and modify the address (before the token) from "http://86fc36e2f382:8888" to "http://localhost:8088/" leaving the token part intact.



## Building the latest paddle and pushing an image

1. Build the latest-dev container:  docker run -v `pwd`:/paddle nguyenthuan/paddle:latest-dev
// this is not required: Getting inside the container: docker run  -it -v `pwd`:/paddle nguyenthuan/paddle:latest-dev /bin/bash
2. docker login
3. cd build
4. Building a new image: docker build -t sidgoyal78/paddle:benchmark11042018  .  (this command isn't fully correct: this results in an image with an unknown tag, so we need to rename it using: docker tag e2a93689725a sidgoyal78/paddle:benchmark12042018)
5. Pushing it to dockerhub: docker push  sidgoyal78/paddle:benchmark11042018 

