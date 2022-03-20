import wandb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import tensorflow as tf
from wandb.keras import WandbCallback
from pkg_resources import resource_filename
from ml_classifier.nn import prepare_data, get_config
from keras.constraints import maxnorm
from sklearn.metrics import classification_report


def model():
    model = Sequential()
    model.add(Dense(100, input_dim=135, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(80, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(20, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(4, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(1, activation='sigmoid'))
    return model


if __name__ == "__main__":
    # initializing wandb with your project name
    run = wandb.init(project="my-nn-project", entity="wenxu",
                     config={'method': 'bayes',
                             'learning_rate': 0.001,
                             'epochs': 100,
                             'batch_size': 64,
                             'loss_function': 'binary_crossentropy'
                             })
    config = wandb.config
    # prepare data
    Config = get_config('/../config/config.yaml')
    neg_path = resource_filename(__name__, Config['neg_feature_file_path']['path'])
    pos_path = resource_filename(__name__, Config['pos_feature_file_path']['path'])
    X_train, X_test, y_train, y_test = prepare_data(neg_path, pos_path)
    # initializing model
    tf.keras.backend.clear_session()
    model = model()
    model.summary()

    # compile model
    optimizer = tf.keras.optimizers.Adam(config.learning_rate)
    model.compile(optimizer, config.loss_function, metrics=['accuracy'])
    _ = model.fit(X_train, y_train, epochs=config.epochs, batch_size=config.batch_size, callbacks=[WandbCallback()])
    y_pred = model.predict(X_test)
    y_pred_bool = []
    for i, predicted in enumerate(y_pred):
        if predicted[0] > 0.5:
            y_pred_bool.append(1)
        else:
            y_pred_bool.append(0)

    print(classification_report(y_test, y_pred_bool))
    loss, accuracy = model.evaluate(X_test, y_test)
    print('Test error rate: ', round((1 - accuracy) * 100, 2))
    wandb.log({'Test Error Rate': round((1 - accuracy) * 100, 2)})
    run.join()
