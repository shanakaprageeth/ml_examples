# setting google cloud suite in ubuntu
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get install apt-transport-https ca-certificates
sudo apt install curl
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update && sudo apt-get install google-cloud-sdk
sudo apt-get install google-cloud-sdk-app-engine-python google-cloud-sdk-app-engine-python-extras
gcloud init
# select your project to init the project

# create a bucket
gsutil mb -l $REGION gs://$BUCKET_NAME

# clone example
git clone --depth 1 https://github.com/GoogleCloudPlatform/cloudml-samples
cd cloudml-samples/census/tf-keras
# install required package in virtual env
pip install --user -r requirements.txt

# train model locally
gcloud ai-platform local train   --package-path trainer   --module-name trainer.task   --job-dir local-training-output

# train model remotely
gcloud ai-platform jobs submit training $JOB_NAME   --package-path trainer/   --module-name trainer.task   --region $REGION   --python-version 3.5   --runtime-version 1.13   --job-dir $JOB_DIR   --stream-logs

# remote hyper parameter training
gcloud ai-platform jobs submit training ${JOB_NAME}_hpt   --config hptuning_config.yaml   --package-path trainer/   --module-name trainer.task   --region $REGION   --python-version 3.5   --runtime-version 1.13   --job-dir $JOB_DIR   --stream-logs

pip3 install pandas

# creating prefiltered input data for prediction
python3 prefilter.py

# model definition for prediction
gcloud ai-platform models create $MODEL_NAME   --regions $REGION
gcloud ai-platform versions create $MODEL_VERSION   --model $MODEL_NAME   --runtime-version 1.13   --python-version 3.5   --framework tensorflow   --origin $SAVED_MODEL_PATH
# predicting the filtered input
gcloud ai-platform predict --model $MODEL_NAME --version $MODEL_VERSION --json-instances prediction_input.json


#clean up
# Delete model version resource
gcloud ai-platform versions delete $MODEL_VERSION --quiet --model $MODEL_NAME

# Delete model resource
gcloud ai-platform models delete $MODEL_NAME --quiet

# Delete Cloud Storage objects that were created
gsutil -m rm -r $JOB_DIR

# If training job is still running, cancel it
gcloud ai-platform jobs cancel $JOB_NAME --quiet