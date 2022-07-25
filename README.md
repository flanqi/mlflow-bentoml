# MLFlow / BentoML Walk-through
## Set Up
### Create Virtual Environment
Create virtual environment using [venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
```bash
python -m venv env # new env created at env/
source env/bin/activate # activate new env
pip install -r requirements.txt
# deactivate 
```

Create conda enviroment (Recommended)
```bash
conda env create -f conda.yaml
conda activate ml-deploy # name of env specified in yml
# conda deactivate
```

## MLFlow Example
The example spins up a chat bot using the pretrained [DialoGPT](https://huggingface.co/microsoft/DialoGPT-medium#:~:text=DialoGPT%20is%20a%20SOTA%20large,single%2Dturn%20conversation%20Turing%20test.) model. One can also finetune it with additional data ([example](https://colab.research.google.com/drive/15wa925dj7jvdvrz8_z3vU7btqAFQLVlG)). An example of finetuning is included in this repo and the model pipeline orchestration script can be found in `run.py`, and this example uses a chat history from personal Messenger downloaded from Facebook to finetune the chat bot and make it sound more like "me".

To run the model "training" with MLFlow tracking, `cd` into the mlflow folder and run:
```bash
python chatbot.py
```
Note that since the model is not a standard sklearn model, we have to create a `mlflow.pyfunc.PythonModel` class with a `predict` method that returns the prediction.

Of course, MLFlow is a lot more powerful with its parameters and metrics tracking capabilities ([example](https://mlflow.org/docs/latest/tutorials-and-examples/tutorial.html)).

Once the model is trained, we can compare models using local UI:
```bash
mlflow ui
```
But in this case, since we did not log any metrics or parameters, this is just to showcase how UI could be potentially used.

### Package Training in Conda Environment
The following command replicates a conda environment specified in the project and runs the model training. It requires the MLProject file to exist.
```bash
mlflow run . --experiment-name chatbot
```

### MLFlow Local Serving
Test serving model locally (find your best model and its model_path in the MLFlow UI):
```bash
mlflow models serve -m MODEL_PATH -p 1234
```

Try a post command:
```bash
curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["text"],"data":[["How are you doing?"]]}' http://127.0.0.1:1234/invocations
```

### Deployment
Although we could not deploy the model to production, it is essential to be able to build a docker image that wraps our model. Below is the command to build a docker image named `my-docker-image`
```bash
mlflow models build-docker \
  -m MODEL_PATH \
  -n my-docker-image \
  --enable-mlserver
```
Once the image is built, we can interact with it by exposing its service endpoint (set to port 8080 in the container by default) locally:
```bash
docker run -p LOCAL_PORT:8080 --name ml-deploy-test my-docker-image
```
Now you may talk to the chat bot the same way as before:
```bash
curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["text"], "data":[["How are you doing?"]]}' http://0.0.0.0:LOCAL_PORT/invocations
```
Once the container is running, we can also login into the container by:
```bash
docker exec -it CONTAINER_ID /bin/sh
```
And the model files are located in /opt/ml/model folder within the container.

## BentoML Example
Instead of a `mlflow.pyfunc.PythonModel`, BentoML uses what's called a `Runnable`. And instead of defining the `predict` function, you can name a couple of functions.

First, we need to install bentoml separately using its pre-release version:
```
pip install --pre bentoml
```
`cd` into the bentoml folder and make sure you have the `service.py` file, and run:
```bash
bentoml serve service:svc --reload
```
This spins up the service, and unlike MLFlow, it also automatically spins up a Swagger UI so you can get predictions/responses using an UI in addition to curl commands.

Curl Command:
```bash
curl \
  -X POST \
  -H "content-type: application/json" \
  --data "How are you?" \
  http://127.0.0.1:3000/talk
```
### Build a Bento
To build a bento (a distribution format containing all necessary files for the service), first create `bentofile.yaml`.

Next, run:
```bash
bentoml build
```
Now, we can serve the model via:
```bash
bentoml serve chatbot:latest --production
```
To generate a docker image containing the Bento:
```bash
bentoml containerize chatbot:latest
```