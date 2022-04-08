import os

import tensorflow as tf
from tensorflow_addons.metrics import F1Score
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import set_seeds, load_data, get_argument_parser


def prepare_data(cfg):
    
    train_df = load_data(cfg=cfg, split_type='train', cached=cfg['cached'])
    dev_df = load_data(cfg=cfg, split_type='dev', cached=cfg['cached'])
    test_df = load_data(cfg=cfg, split_type='test', cached=cfg['cached'])

    lb = LabelBinarizer()
    y_train = lb.fit_transform(train_df['label'])
    y_dev = lb.transform(dev_df['label'])
    y_test = lb.transform(test_df['label'])

    # bag of word
    bow_model = TfidfVectorizer(ngram_range=(1, 2), min_df=3)
    X_train = bow_model.fit_transform(train_df['cleaned_text'])
    X_dev = bow_model.transform(dev_df['cleaned_text'])
    X_test = bow_model.transform(test_df['cleaned_text'])

    # initial feature selection
    k_best = SelectKBest(chi2, k=10)
    X_train = k_best.fit_transform(X_train, y_train)
    X_dev = k_best.transform(X_dev)
    X_test = k_best.transform(X_test)

    return X_train, X_dev, X_test, y_train, y_dev, y_test, lb


def train(cfg, X_train, X_dev, X_test, y_train, y_dev, y_test):

    inputs = tf.keras.layers.Input(shape=(X_train.shape[1],))
    x = tf.keras.layers.Dense(units=256, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(units=128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(units=5, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg['learning_rate'])

    loss = tf.keras.losses.CategoricalCrossentropy()

    f1_score = F1Score(num_classes=5, average='weighted', 
                       name='weighted_f1_score')

    file_path = os.path.join(cfg['checkpoint_dir'], 'task1', 'best_model')
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=file_path, 
                                                      monitor='val_loss', 
                                                      verbose=1,
                                                      save_best_only=True)
    early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                     min_delta=0, 
                                                     patience=5, verbose=2)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                        factor=0.2, 
                                                        patience=3, verbose=1)

    model.compile(optimizer=optimizer, loss=loss, metrics=[f1_score])

    model.fit(x=X_train.todense(), y=y_train, epochs=cfg['epochs'], verbose=1,
              shuffle=True, callbacks=[early_stopper, lr_scheduler, 
              checkpointer], validation_data=(X_dev.todense(), y_dev))

    model.load_weights(file_path)  # load the best model
    model.evaluate(x=X_test.todense(), y=y_test)

    return  # variables for further error analysis


if __name__ == '__main__':

    cfg = get_argument_parser().parse_args().__dict__
    
    set_seeds(cfg)

    X_train, X_dev, X_test, y_train, \
        y_dev, y_test, label_binarizer = prepare_data(cfg)

    train(cfg, X_train, X_dev, X_test, y_train, y_dev, y_test)