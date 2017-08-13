# MNIST Tensorflow Starter Code

한국어 버전 설명은 다음 링크를 따라가주세요: [Link](README.md)

This repo contains starter code for training and evaluating machine learning
models over the mnist dataset.

The code gives an end-to-end working example for reading the dataset, training a
TensorFlow model, and evaluating the performance of the model. Out of the box,
you can train several [model architectures](#overview-of-models) over features.
The code can easily be extended to train your own custom-defined models.

It is possible to train and evaluate on mnist in two ways: on Google Cloud
or on your own machine. This README provides instructions for both.

## Table of Contents
* [Running on Google's Cloud Machine Learning Platform](#running-on-googles-cloud-machine-learning-platform)
   * [Requirements](#requirements)
   * [Testing Locally](#testing-locally)
   * [Training on the Cloud](#training-on-features)
   * [Evaluation and Inference](#evaluation-and-inference)
   * [Inference Using Batch Prediction](#inference-using-batch-prediction)
   * [Accessing Files on Google Cloud](#accessing-files-on-google-cloud)
   * [Using Larger Machine Types](#using-larger-machine-types)
* [Running on Your Own Machine](#running-on-your-own-machine)
   * [Requirements](#requirements-1)
* [Overview of Models](#overview-of-models)
* [Overview of Files](#overview-of-files)
   * [Training](#training)
   * [Evaluation](#evaluation)
   * [Inference](#inference)
   * [Misc](#misc)
* [TODO for participants](#todo-for-participants)
* [Etc](#etc)
* [About This Project](#about-this-project)

## Running on Google's Cloud Machine Learning Platform

### Requirements

This option requires you to have an appropriately configured Google Cloud
Platform account. To create and configure your account, please make sure you
follow the instructions [here](https://cloud.google.com/ml/docs/how-tos/getting-set-up).

Please also verify that you have Python 2.7+ and Tensorflow 1.0.1 or higher
installed by running the following commands:

```sh
python --version
python -c 'import tensorflow as tf; print(tf.__version__)'
```

### Testing Locally
All gcloud commands should be done from the directory *immediately above* the
source code. You should be able to see the source code directory if you
run 'ls'.

As you are developing your own models, you will want to test them
quickly to flush out simple problems without having to submit them to the cloud.
You can use the `gcloud beta ml local` set of commands for that.
Here is an example command line:

```sh
gcloud ml-engine local train \
--package-path=mnist --module-name=mnist.train -- \
--train_data_pattern='gs://kmlc_test_train_bucket/mnist/train.tfrecords' \
--train_dir=/tmp/kmlc_mnist_train --model=LogisticModel --start_new_model
```

You might want to download the training data files to the current directory.

```sh
gsutil cp gs://kmlc_test_train_bucket/mnist/train.tfrecords .
```

Once you download the files, you can point the job to them using the
'train_data_pattern' argument (i.e. instead of pointing to the "gs://..."
files, you point to the local files).

Once your model is working locally, you can scale up on the Cloud
which is described below.

### Training on the Cloud

You'll use Google Cloud to access the training and test files. You'll also be given free credits to try out Google Cloud. Below are some step-by-step tutorials to set up and get data, and submit training/testing jobs to Google Cloud ML.

#### Set up your Google Cloud project

1. Create a new Cloud Platform project. This is where your project lives.
   - Click Create Project and follow instructions.
   - Enable billing for your project. This links a billing method to your project. For a new account, you will already have $300 in trial credits within your default billing account.
   - [Enable the APIs](https://console.cloud.google.com/flows/enableapi?apiid=ml.googleapis.com,dataflow,compute_component,logging,storage_component,storage_api,bigquery) but ignore adding Credentials. This enables the set of Cloud APIs that are needed for Cloud ML functionality such as Cloud Logging to get your training logs. Other APIs include: Cloud Machine Learning, Dataflow, Compute Engine, Cloud Storage, Cloud Storage JSON, and BigQuery APIs.
   - You may have to select the newly-created project.
   - Expect this to take some time.
   - After the APIs are enabled, do not “Go to Credentials”
2. Set up your environment using cloud shell
   - There are three paths to use Google Cloud for this competition: Cloud shell, local (Mac/Linux), & Docker. To start we recommend the cloud shell to avoid having to set up a local environment.
   - Before you click the cloud shell button, make sure that you have already selected your newly-created project (in the screenshot example, the project name is My First Project)
   - You can start a cloud shell by clicking on the cloud shell icon shown in the screenshot below.
     <img src="https://codelabs.developers.google.com/codelabs/cpb100-cloud-sql/img/d91c4415ec90a651.png" />
   - You should run all of the following commands inside of the cloud shell command line.
   - The first step to setting up the environment is to configure the gcloud command-line tool to use your selected project, where [selected-project-id] is your project id, without the enclosing brackets.
     ``` sh
     gcloud config set project [selected-project-id]
     ```
   - Python version should be 2.7+
     ``` sh
     $ python --version
     Python 2.7.9
     ```
   - Install the latest version of TensorFlow (1.2.1) with the following 2 command lines.
     ```sh
     pip download tensorflow
     pip install --user -U tensorflow*.whl
     ```
3. Verify the Google Cloud SDK Components
   - List the models to verify that the command returns an empty list.
   ```sh
   gcloud ml-engine models list
   ```
   - The command will an empty list, and after you start creating models, you can see them listed by using this command.

#### Running training

The following commands will train a model on Google Cloud. Following commands are need to be executed on Google Cloud Shell.

```sh
BUCKET_NAME=gs://${USER}_kmlc_mnist_train_bucket
# (One Time) Create a storage bucket to store training logs and checkpoints.
gsutil mb -l us-east1 $BUCKET_NAME
# Submit the training job.
JOB_NAME=kmlc_mnist_train_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=mnist --module-name=mnist.train \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=mnist/cloudml-gpu.yaml \
-- --train_data_pattern='gs://kmlc_test_train_bucket/mnist/train.tfrecords' \
--model=LogisticModel \
--train_dir=$BUCKET_NAME/kmlc_mnist_train_logistic_model
```

In the 'gsutil' command above, the 'package-path' flag refers to the directory
containing the 'train.py' script and more generally the python package which
should be deployed to the cloud worker. The module-name refers to the specific
python script which should be executed (in this case the train module).

It may take several minutes before the job starts running on Google Cloud.
When it starts you will see outputs like the following:

```
training step 270| Hit@1: 0.68 PERR: 0.52 Loss: 638.453
training step 271| Hit@1: 0.66 PERR: 0.49 Loss: 635.537
training step 272| Hit@1: 0.70 PERR: 0.52 Loss: 637.564
```

At this point you can disconnect your console by pressing "ctrl-c". The
model will continue to train indefinitely in the Cloud. Later, you can check
on its progress or halt the job by visiting the
[Google Cloud ML Jobs console](https://console.cloud.google.com/ml/jobs).

You can train many jobs at once and use tensorboard to compare their performance
visually.

```sh
tensorboard --logdir=$BUCKET_NAME --port=8080
```

Once tensorboard is running, you can access it at the following url:
[http://localhost:8080](http://localhost:8080).
If you are using Google Cloud Shell, you can instead click the Web Preview button
on the upper left corner of the Cloud Shell window and select "Preview on port 8080".
This will bring up a new browser tab with the Tensorboard view.

### Evaluation and Inference
Here's how to evaluate a model on the validation dataset:

```sh
JOB_TO_EVAL=kmlc_mnist_train_logistic_model
JOB_NAME=kmlc_mnist_eval_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=mnist --module-name=mnist.eval \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=mnist/cloudml-gpu.yaml \
-- --eval_data_pattern='gs://kmlc_test_train_bucket/mnist/validation.tfrecords' \
--model=LogisticModel \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} --run_once=True
```

And here's how to perform inference with a model on the test set:

```sh
JOB_TO_EVAL=kmlc_mnist_train_logistic_model
JOB_NAME=kmlc_mnist_inference_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=mnist --module-name=mnist.inference \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=mnist/cloudml-gpu.yaml \
-- --input_data_pattern='gs://kmlc_test_train_bucket/mnist/test.tfrecords' \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
--output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions.csv
```

Note the confusing use of 'training' in the above gcloud commands. Despite the
name, the 'training' argument really just offers a cloud hosted
python/tensorflow service. From the point of view of the Cloud Platform, there
is no distinction between our training and inference jobs. The Cloud ML platform
also offers specialized functionality for prediction with
Tensorflow models, but discussing that is beyond the scope of this readme.

Once these job starts executing you will see outputs similar to the
following for the evaluation code:

```
examples_processed: 1024 | global_step 447044 | Batch Hit@1: 0.782 | Batch PERR: 0.637 | Batch Loss: 7.821 | Examples_per_sec: 834.658
```

and the following for the inference code:

```
num examples processed: 8192 elapsed seconds: 14.85
```

### Inference Using Batch Prediction
To perform inference faster, you can also use the Cloud ML batch prediction
service.

First, find the directory where the training job exported the model:

```
gsutil list ${BUCKET_NAME}/kmlc_mnist_train_logistic_model/export
```

You should see an output similar to this one:

```
${BUCKET_NAME}/kmlc_mnist_train_logistic_model/export/
${BUCKET_NAME}/kmlc_mnist_train_logistic_model/export/step_1/
${BUCKET_NAME}/kmlc_mnist_train_logistic_model/export/step_1001/
${BUCKET_NAME}/kmlc_mnist_train_logistic_model/export/step_2001/
${BUCKET_NAME}/kmlc_mnist_train_logistic_model/export/step_3001/
```

Select the latest version of the model that was saved. For instance, in our
case, we select the version of the model that was saved at step 3001:

```
EXPORTED_MODEL_DIR=${BUCKET_NAME}/kmlc_mnist_train_logistic_model/export/step_3001/
```

Start the batch prediction job using the following command:

```
JOB_NAME=kmlc_mnist_batch_predict_$(date +%Y%m%d_%H%M%S); \
gcloud ml-engine jobs submit prediction ${JOB_NAME} --verbosity=debug \
--model-dir=${EXPORTED_MODEL_DIR} --data-format=TF_RECORD \
--input-paths='gs://kmlc_test_train_bucket/mnist/test.tfrecords' \
--output-path=${BUCKET_NAME}/batch_predict/${JOB_NAME}.csv --region=us-east1 \
--runtime-version=1.2 --max-worker-count=10
```

You can check the progress of the job on the
[Google Cloud ML Jobs console](https://console.cloud.google.com/ml/jobs). To
have the job complete faster, you can increase 'max-worker-count' to a
higher value.

Once the batch prediction job has completed, turn its output into a submission
in the CVS format by running the following commands:

```
# Copy the output of the batch prediction job to a local directory
mkdir -p /tmp/batch_predict/${JOB_NAME}
gsutil -m cp -r ${BUCKET_NAME}/batch_predict/${JOB_NAME}.csv/* /tmp/batch_predict/${JOB_NAME}.csv
```

Submit the resulting file /tmp/batch_predict/${JOB_NAME}.csv to Kaggle.

### Accessing Files on Google Cloud

You can browse the storage buckets you created on Google Cloud, for example, to
access the trained models, prediction CSV files, etc. by visiting the
[Google Cloud storage browser](https://console.cloud.google.com/storage/browser).

Alternatively, you can use the 'gsutil' command to download the files directly.
For example, to download the output of the inference code from the previous
section to your local machine, run:


```
gsutil cp $BUCKET_NAME/${JOB_TO_EVAL}/predictions.csv .
```


### Using Larger Machine Types
Some complex models can take as long as a week to converge when
using only one GPU. You can train these models more quickly by using more
powerful machine types which have additional GPUs. To use a configuration with
4 GPUs, replace the argument to `--config` with `mnist/cloudml-4gpu.yaml`.
Be careful with this argument as it will also increase the rate you are charged
by a factor of 4 as well.

## Running on Your Own Machine

### Requirements

The starter code requires Tensorflow. If you haven't installed it yet, follow
the instructions on [tensorflow.org](https://www.tensorflow.org/install/).
This code has been tested with Tensorflow 1.0.1. Going forward, we will continue
to target the latest released version of Tensorflow.

Please verify that you have Python 2.7+ and Tensorflow 1.0.1 or higher
installed by running the following commands:

```sh
python --version
python -c 'import tensorflow as tf; print(tf.__version__)'
```

Downloading files
``` sh
gsutil cp gs://kmlc_test_train_bucket/mnist/train* .
gsutil cp gs://kmlc_test_train_bucket/mnist/test* .
gsutil cp gs://kmlc_test_train_bucket/mnist/validation* .
```

Training
```sh
python train.py --train_data_pattern='/path/to/training/files/*' --train_dir=/tmp/mnist_train --model=LogisticModel --start_new_model
```

Validation
```sh
python eval.py --eval_data_pattern='/path/to/validation/files' --train_dir=/tmp/mnist_train --model=LogisticModel --run_once=True
```

Generating submission
```sh
python inference.py --output_file=/path/to/predictions.csv --input_data_pattern='/path/to/test/files/*' --train_dir=/tmp/mnist_train
```

## Overview of Models

This sample code contains implementation of the logistic model:

*   `LogisticModel`: Linear projection of the output features into the label
                     space, followed by a sigmoid function to convert logit
                     values to probabilities.

## Overview of Files

### Training
*   `train.py`: Defines the parameters and procedures for training. You can modify the parameters such as the location of training dataset, the model to be used for training, the batch size, the loss function to be used, the learning rate, etc. Depending on the model, you may want to modify get_input_data_tensors() on how data is shuffled.
*   `losses.py`: Defines the loss functions. You can call train.py to use any of the loss functions defined in losses.py.
*   `models.py`: Contains the base class for defining a model.
*   `mnist_models.py`: Contains the definition for models that that take the aggregated features as input, and you should add your own models here. You can invoke any model by calling train.py using --model=YourModelName.
*   `export_model.py`: Provides a class to export a model during training for later use in batch prediction.
*   `readers.py`: Contains the definition of the dataset, and describes how input data are prepared. You can preprocess the input files by modifying prepare_serialized_examples(). For example, you may want to resize the data or introduce some random noise.

### Evaluation
*   `eval.py`: The primary script for evaluating models. Once the model is trained, you can call eval.py with the --train_dir=/path/to/model and --model=YourModelName to load your model from the files in train_dir. Most likely you do not need to modify this file.
*   `eval_util.py`: Provides a class that calculates all evaluation metrics.
*   `average_precision_calculator.py`: Functions for calculating average precision.
*   `mean_average_precision_calculator.py`: Functions for calculating mean average precision.

### Inference
*   `inference.py`: Generates an output file containing predictions of the model over a set of data. Call inference.py on the test data to generate a list of predicted labels. For the supervised learning problems, the evaluation is based on the accuracy of the predicted labels.

### Misc
*   `README.md`: This documentation.
*   `utils.py`: Common functions.
*   `convert_prediction_from_json_to_csv.py`: Converts the JSON output of batch prediction into a CSV file for submission.

## TODO for participants
* Create your model in mnist_models.py.
* Create a loss function in losses.py.
* If necessary, add input preprocessing in readers.py.
* Adjust the parameters in train.py, such as the batch size and learning rate, and modify the training procedure if it is necessary.
* Train your model by calling train.py using your model name and loss function.
* Call eval.py to examine the performance of trained model on validation data.
* Call inference.py to obtain the predicted labels using your model, and submit these labels in Kaggle for evaluation.
* In general, you are free to modify any file if it improves the performance of your model.

## Etc
* [Tensorflow MNIST Tutorial](https://www.tensorflow.org/get_started/mnist/beginners)

## About This Project
This project is meant help people quickly get started working with the
[mnist](LINK_TO_KMLC_SITE) dataset.
This is not an official Google product.
