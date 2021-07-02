import logging
import warnings
from build_model import BuildModel
from preprocessing import preprocessing, get_dataset, get_spectrogram
import mlflow
import argparse
import platform
import requests

import azureml.core
from azureml.core import Workspace, Experiment
import tensorflow as tf
import tensorflow.keras as keras

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def plot_accuracy(history):
    train_accuracies = history["accuracy"]
    val_accuracies = history["val_accuracy"]
    plt.plot(train_accuracies, '-bx')
    plt.plot(val_accuracies, "-rx")
    plt.margins(0.05)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(["Training", "Validation"])
    plt.title('Accuracy vs. No. of epochs')


def plot_recall(history):
    train_accuracies = history["recall"]
    val_accuracies = history["val_recall"]
    plt.plot(train_accuracies, '-bx')
    plt.plot(val_accuracies, "-rx")
    plt.margins(0.05)
    plt.xlabel('epoch')
    plt.ylabel('recall')
    plt.legend(["Training", "Validation"])
    plt.title('Recall vs. No. of epochs')


def plot_precision(history):
    train_accuracies = history["precision"]
    val_accuracies = history["val_precision"]
    plt.plot(train_accuracies, '-bx')
    plt.plot(val_accuracies, "-rx")
    plt.margins(0.05)
    plt.xlabel('epoch')
    plt.ylabel('precision')
    plt.legend(["Training", "Validation"])
    plt.title('Precision vs. No. of epochs')


def plot_auc(history):
    train_accuracies = history["auc"]
    val_accuracies = history["val_auc"]
    plt.plot(train_accuracies, '-bx')
    plt.plot(val_accuracies, "-rx")
    plt.xlabel('epoch')
    plt.margins(0.05)
    plt.ylabel('auc')
    plt.legend(["Training", "Validation"])
    plt.title(' Area Under the ROC Curve vs. No. of epochs')


def plot_losses(history):
    train_losses = history["loss"]
    val_losses = history["val_loss"]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.margins(0.05)
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')


def plot_historys(history):
    fig = plt.figure(figsize=(14, 7))

    plt.subplot(2, 2, 1)
    plot_precision(history)

    plt.subplot(2, 2, 2)
    plot_losses(history)

    plt.subplot(2, 2, 3)
    plot_recall(history)

    plt.subplot(2, 2, 4)
    plot_auc(history)

    plt.subplot(3, 2, 5)
    plot_accuracy(history)


    fig.tight_layout()
    figure_file = "./figures/train_val_history.png"
    plt.savefig(figure_file)
    return figure_file


class Trainer:

    def __init__(self, experiment_name, data_path, output_shape, subscription_id, workspace_name, resource_group, tracking_uri=None, run_origin="none"):
        self.experiment_name = experiment_name
        self.data_path = data_path
        self.run_origin = run_origin
        self.tracking_uri = tracking_uri
        self.subscription_id = subscription_id
        self.workspace_name = workspace_name
        self.resource_group = resource_group


        print(f"experiment_name: {self.experiment_name}")
        print(f"run_origin: {run_origin}")
        print(f"data_path: {self.data_path}")
        
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = preprocessing(self.data_path)
        print(f"\nSize of train set: {len(self.X_train)}")
        print(f"Size of validation set: {len(self.X_val)}")
        print(f"Size of test set: {len(self.X_test)}")
        self.input_shape = self.X_train.shape[1:]
        self.output_shape = output_shape

        self.y_train = keras.utils.to_categorical(
            self.y_train, num_classes=self.output_shape, dtype="float32")
        self.y_val = keras.utils.to_categorical(
            self.y_val, num_classes=self.output_shape, dtype="float32")
        self.y_test = keras.utils.to_categorical(
            self.y_test, num_classes=self.output_shape, dtype="float32")
        
        if (self.experiment_name != "none"):
            # ws = Workspace(subscription_id=self.subscription_id,
            #                 resource_group=self.resource_group, workspace_name=self.workspace_name)
            # mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
            # try:
                # self.experiment_id = mlflow.create_experiment(self.experiment_name)
            # except:
            # mlflow.set_experiment(self.experiment_name)
            self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id

            # print(f"experiment_id: {self.experiment_id}")

    def train_model(self, model, **kwargs):
        

        with mlflow.start_run(run_name=self.experiment_name) as run:
            run_id = run.info.run_uuid
            experimentID = run.info.experiment_id
            print(f"run_id: {run_id}")

            experiment_id = run.info.experiment_id

            print(f"experiment_id: {experiment_id}")

            metrics = [
                keras.metrics.CategoricalAccuracy(name="accuracy"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.AUC(name="auc")
            ]

            optimizer = keras.optimizers.Adam(
                learning_rate=kwargs["learning_rate"])

            model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=metrics)

            mc = keras.callbacks.ModelCheckpoint(
                filepath="models/best_model.h5",
                monitor="accuracy",
                mode="max",
                verbose=1,
                save_best_only=True
            )
            mlflow.keras.autolog()
            history = model.fit(
                self.X_train, self.y_train, 
                validation_data=(self.X_val, self.y_val),
                epochs=kwargs["epochs"],
                batch_size=kwargs["batch_size"],
                verbose=1,
                callbacks=[mc]
            )

            model.save("./models/final_model.h5")
            
            loss, accuracy, precision, recall, auc = model.evaluate(self.X_test, self.y_test, verbose=2)

            figure_file = plot_historys(history.history)

            # log dict
            mlflow.log_dict(history.history, "history.json")
            # log parameters
            mlflow.log_param("input", str(self.input_shape))
            mlflow.log_param("hidden_layers", len(model.layers)-2)
            mlflow.log_param("output", self.output_shape)
            mlflow.log_param("epochs", kwargs["epochs"])
            mlflow.log_param("loss_function", "categorical_crossentropy")
            mlflow.log_param("optimizer", "Adam")
            
            # log metrics
            mlflow.log_metric("test_loss", loss)
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_precision", precision)
            mlflow.log_metric("test_recall", recall)
            mlflow.log_metric("test_auc", auc)

            # log model
            best_model = keras.models.load_model("./models/best_model.h5")
            mlflow.keras.log_model(best_model, "final_model")
            mlflow.keras.log_model(model, "best_model")
            
            # log artifacts
            mlflow.log_artifact(figure_file)
            
            # set tags
            mlflow.set_tag("data_path", self.data_path)
            mlflow.set_tag("experimet_id", experiment_id)
            mlflow.set_tag("experiment_name", self.experiment_name)
            mlflow.set_tag("run_origin", self.run_origin)
            mlflow.set_tag("platform", platform.system())


def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(
        description="Music Genre Experiment")

    parser.add_argument("--data_path", action="store", required=True,
                        help="Path of waveform dataset ", dest="data_path")

    parser.add_argument("--experiment_name", action="store", required=True,
                        help="Name for MLflow experiment", dest="experiment_name")

    parser.add_argument("--subscription_id", action="store", required=True,
                        help="Subscription id", dest="subscription_id")

    parser.add_argument("--resource_group", action="store", required=True,
                        help="Resource group", dest="resource_group")

    parser.add_argument("--workspace_name", action="store", required=True,
                        help="Workspace name", dest="workspace_name")

    parser.add_argument("--height", action="store", required=True,
                        help="Height of spectrogram", dest="height")

    parser.add_argument("--width", action="store", required=True,
                    help="Width of spectrogram", dest="width")
    
    parser.add_argument("--channels", action="store", required=True,
                    help="Channels of spectrogram", dest="channels")

    parser.add_argument("--output_shape", action="store", required=True,
                        help="Chose what ml algorithm for apply", dest="output_shape")

    parser.add_argument("--run_origin", action="store", required=False,
                        help="Choose run origin", dest="run_origin")

    arguments = parser.parse_args()

    data_path = arguments.data_path
    experiment_name = arguments.experiment_name
    subscription_id = arguments.subscription_id
    resource_group = arguments.resource_group
    workspace_name = arguments.workspace_name
    run_origin = arguments.run_origin

    height, width, channels = int(arguments.height), int(arguments.width), int(arguments.channels)
    input_shape = (height, width, channels)
    output_shape = int(arguments.output_shape)

    bd = BuildModel(input_shape=input_shape, output_shape=output_shape)
    model = bd.feed_foward()
    model.summary()
    trainer = Trainer(experiment_name=experiment_name, data_path=data_path, output_shape=output_shape, 
        subscription_id=subscription_id, resource_group=resource_group, workspace_name=workspace_name, run_origin=run_origin)
    trainer.train_model(model, epochs=100, learning_rate=1e-3, batch_size=256)


if __name__ == '__main__':
    main()
    
