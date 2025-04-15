SSH into TACC and then into Classroom VM

Then run:

docker pull arvindvinod/hurricane-damage-classifier:latest  

docker-compose up  


Now you can test the API with:


curl http://localhost:5000/summary  


Then grading scripts can be ran
