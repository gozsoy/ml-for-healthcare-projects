import os

import tensorflow as tf
from tensorflow_addons.metrics import F1Score
from sklearn.preprocessing import LabelBinarizer
from transformers import TFAutoModel, AutoTokenizer

from utils import set_seeds, load_data, get_argument_parser


class BertBasedClassifier(tf.keras.Model):
    def __init__(self, fine_tune=False):
        super().__init__()

        self.bert_base = TFAutoModel.from_pretrained("emilyalsentzer/ \
                                                     Bio_ClinicalBERT", 
                                                     from_pt=True)

        if not fine_tune:
            for layer in self.bert_base.layers:
                layer.trainable = False

        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=5, activation='softmax')

    def call(self, x):
 
        output = self.bert_base(input_ids=x[0], attention_mask=x[1])

        last_hidden_state = output[0]
        cls = last_hidden_state[:, 0, :]

        output = self.dense1(cls)
        output = self.dense2(output)
        
        return output


def text2bertToken(tokenizer, text, y):

    inputs = tokenizer(text=text, max_length=2, padding=True, 
                       truncation=True, return_tensors='tf')
    ds = tf.data.Dataset.from_tensor_slices(((inputs['input_ids'],
                                              inputs['attention_mask']), y))
    ds = ds.shuffle(1000).batch(64)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def prepare_data(cfg):

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/ \
                                              Bio_ClinicalBERT")

    train_df = load_data(cfg=cfg, split_type='train', cached=cfg['cached'])
    dev_df = load_data(cfg=cfg, split_type='dev', cached=cfg['cached'])
    test_df = load_data(cfg=cfg, split_type='test', cached=cfg['cached'])

    lb = LabelBinarizer()
    y_train = lb.fit_transform(train_df['label'])
    y_dev = lb.transform(dev_df['label'])
    y_test = lb.transform(test_df['label'])

    train_ds = text2bertToken(tokenizer, list(train_df['cleaned_text'].values),
                              y_train)
    dev_ds = text2bertToken(tokenizer, list(dev_df['cleaned_text'].values), 
                            y_dev)
    test_ds = text2bertToken(tokenizer, list(test_df['cleaned_text'].values),
                             y_test)

    return train_ds, dev_ds, test_ds, lb


def train(cfg, train_ds, dev_ds, test_ds):

    model = BertBasedClassifier(fine_tune=cfg['finetune_bert'])

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg['learning_rate'])

    loss = tf.keras.losses.CategoricalCrossentropy()

    f1_score = F1Score(num_classes=5, average='weighted',
                       name='weighted_f1_score')

    file_path = os.path.join(cfg['checkpoint_dir'], 'task3', 'best_model')
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

    model.fit(x=train_ds, epochs=cfg['epochs'], verbose=1,
              shuffle=True, callbacks=[early_stopper, lr_scheduler, 
              checkpointer], validation_data=dev_ds)

    model.load_weights(file_path)  # load the best model
    model.evaluate(test_ds)

    return  # variables for further error analysis


if __name__ == '__main__':

    cfg = get_argument_parser().parse_args().__dict__
    
    set_seeds(cfg)

    train_ds, dev_ds, test_ds, label_binarizer = prepare_data(cfg)

    train(cfg, train_ds, dev_ds, test_ds)
