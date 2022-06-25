from wrangl.learn import SupervisedModel, metrics as M
import torch
from torch.nn import functional as F
from transformers import GPT2LMHeadModel as AutoModel, AutoTokenizer

class Model(SupervisedModel):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.lm = AutoModel.from_pretrained(cfg.lm)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.lm)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.acc = M.Accuracy()

    def featurize(self, batch: list):
        """
        Converts a batch of examples to features.
        By default this returns the batch as is.

        Alternatively you may want to set `collate_fn: "ignore"` in your config and use `featurize` to convert raw examples into features.
        """
        for ex in batch:

         correct_sentence = self.tokenizer(
            [ex['sentence'].replace('_', ex["option1" if ex["answer"] == '1' else "option2"]) for ex in batch],
            return_tensors='pt',
            add_special_tokens=True,
            truncation=True,
            padding=True,
            return_attention_mask=True,
            max_length=self.hparams.max_context_length,
        ).to(self.device)
        incorrect_sentence = self.tokenizer(
            [ex['sentence'].replace('_', ex["option1" if ex["answer"] == '2' else "option2"]) for ex in batch],
            return_tensors='pt',
            add_special_tokens=True,
            truncation=True,
            padding=True,
            return_attention_mask=True,
            max_length=self.hparams.max_context_length,
        ).to(self.device)
        return dict(correct_sentence=correct_sentence, incorrect_sentence=incorrect_sentence)

    def compute_loss(self, out, feat, batch):
        return out["loss"]

    def compute_metrics(self, pred: list, gold: list, batch: list = None) -> dict:
        # this returns a dictionary {'val_acc': x}
        return self.acc(pred, gold)

    def extract_context(self, feat, batch):
        return [ex['sentence'] for ex in batch]

    def extract_pred(self, out, feat, batch):
        pred = []
        for p, ex in zip(out, batch):
            pred.append(ex['sentence'].replace('_', ex["option1"] if p == 0 else "option2"))
        return pred

    @classmethod
    def extract_gold(cls, feat, batch):
         return [ex['sentence'].replace('_', ex["option1" if ex["answer"] == '1' else "option2"]) for ex in batch]

    def forward(self, feat, batch):
        correct_out = self.lm(feat['correct_sentence']['input_ids'], attention_mask=feat['correct_sentence']['attention_mask'], labels=feat['correct_sentence']['input_ids'])
        incorrect_out = self.lm(feat['incorrect_sentence']['input_ids'], attention_mask=feat['incorrect_sentence']['attention_mask'], labels=feat['incorrect_sentence']['input_ids'])
        loss = correct_out["loss"] - incorrect_out["loss"]
        return dict(loss=loss, logits=None)

    def infer(self, feat, batch):
        ps = []
        for i in range(len(batch)):
            correct_out = self.lm(feat['correct_sentence']['input_ids'][i], attention_mask=feat['correct_sentence']['attention_mask'][i], labels=feat['correct_sentence']['input_ids'][i])
            incorrect_out = self.lm(feat['incorrect_sentence']['input_ids'][i], attention_mask=feat['incorrect_sentence']['attention_mask'][i], labels=feat['incorrect_sentence']['input_ids'][i])
            loss = correct_out["loss"] - incorrect_out["loss"]
            ps.append(int(loss > 0))  # 0 means first (correct) sentence
        return ps

