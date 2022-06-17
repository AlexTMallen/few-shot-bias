from wrangl.learn import SupervisedModel, metrics as M
import torch
from transformers import AutoModelForSequenceClassification as AutoModel, AutoTokenizer


class Model(SupervisedModel):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.lm = AutoModel.from_pretrained(cfg.lm, num_labels=2)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.lm)
        self.acc = M.Accuracy()

    def featurize(self, batch: list):
        """
        Converts a batch of examples to features.
        By default this returns the batch as is.

        Alternatively you may want to set `collate_fn: "ignore"` in your config and use `featurize` to convert raw examples into features.
        """
        sentence = self.tokenizer(
            [ex['sentence'] + ' choice ' + ex['option1'] + ' choice ' + ex['option2'] for ex in batch],
            return_tensors='pt',
            add_special_tokens=True,
            truncation=True,
            padding=True,
            return_attention_mask=True,
            max_length=self.hparams.max_context_length,
        ).to(self.device)
        mapping = {'1': 0, '2': 1}
        labels = torch.tensor([mapping[ex['answer']] for ex in batch], device=self.device)
        return dict(sentence=sentence, labels=labels)

    def compute_loss(self, out, feat, batch):
        return out.loss

    def compute_metrics(self, pred: list, gold: list, batch: list = None) -> dict:
        # this returns a dictionary {'val_acc': x}
        return self.acc(pred, gold)

    def extract_context(self, feat, batch):
        return [ex['sentence'] for ex in batch]

    def extract_pred(self, out, feat, batch):
        pred = []
        for p, ex in zip(out, batch):
            if p == 0:
                pred.append(ex['option1'])
            elif p == 1:
                pred.append(ex['option2'])
            else:
                raise NotImplementedError('Unknown prediction {} for example {}'.format(p, ex))
        return pred

    @classmethod
    def extract_gold(cls, feat, batch):
        return [ex['option{}'.format(ex['answer'])] for ex in batch]

    def forward(self, feat, batch):
        out = self.lm(feat['sentence']['input_ids'], attention_mask=feat['sentence']['attention_mask'], labels=feat['labels'])
        return out

    def infer(self, feat, batch):
        return self.lm(feat['sentence']['input_ids'], attention_mask=feat['sentence']['attention_mask']).logits.max(1)[1].tolist()
