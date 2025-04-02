#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
from transformers import AutoModel, AutoTokenizer
from model3_v2 import UnicodeClassifier_v3_2
from unique_chars import get_unique_chars
from unicode_character_mapping import idx_to_char_mapping


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    tokenizer = None
    model = None
    idx2char = None
    device = "cpu"
    def __init__(self, device="cpu"):
        if MyModel.tokenizer is None:
            MyModel.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        if MyModel.model is None:
            MyModel.model = UnicodeClassifier_v3_2("xlm-roberta-base", 34060)
        if MyModel.device != device:
            MyModel.device = device
        if MyModel.idx2char is None:
            MyModel.idx2char = idx_to_char_mapping

    @classmethod
    def load_training_data(cls):
        # your code here
        # this particular model doesn't train
        return []

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(cls.tokenizer(inp, return_tensors="pt"))
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # your code here
        pass

    def run_pred(self, data):
        # your code here
        preds = []
        MyModel.model.to(MyModel.device)
        for input in data:
            logits = MyModel.model(input["input_ids"].to(MyModel.device), input["attention_mask"].to(MyModel.device))
            _, top_k_indices = torch.topk(logits.cpu(), k=3, dim=-1)
            top_k_chars = ["".join([MyModel.idx2char[idx.item()] for idx in i]) for i in top_k_indices]
            preds.append(top_k_chars[0])
        return preds

    def save(self, work_dir):
        # your code here
        # model.save_pretrained("./my_model_checkpoint"), model from transformers

        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            f.write('dummy save')

    @classmethod
    def load(cls, work_dir):
        # your code here
        # model = AutoModelForSequenceClassification.from_pretrained("./my_model_checkpoint") model from transformers

        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        """
        mm = UnicodeClassifier_v3_2(model_name, num_classes).to(device)
        mm.load_state_dict(torch.load("/content/v3_2lr=0.0002, batch_size=128.pth"))
        mm.eval()
        """
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.idx2char, _ = get_unique_chars(work_dir)
        cls.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        path = "".join([work_dir, "/mycheckpoint2.pth"])
        cls.model = torch.load(path, weights_only=False, map_location=cls.device)
        cls.model.to(cls.device)
        cls.model.eval()
        return MyModel()


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
