export BUCKET_NAME="test-bucket"
export REGION="us-central1"
export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export BUCKET_NAME="${PROJECT_ID}-test-bucket"
export JOB_NAME="my_first_keras_job"
export JOB_DIR="gs://$BUCKET_NAME/keras-job-dir"

export MODEL_NAME="my_first_keras_model"
export MODEL_VERSION="v1"
export SAVED_MODEL_PATH=$(gsutil ls $JOB_DIR/keras_export | tail -n 1)
