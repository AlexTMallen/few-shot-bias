#!/usr/bin/env python
import os
import hydra
import logging
import ujson as json
from wrangl.learn import SupervisedModel


logger = logging.getLogger(__name__)


def load_data(fname):
    return [json.loads(x) for x in open(fname)]


@hydra.main(config_path='conf', config_name='default')
def main(cfg):
    fcheckpoint = os.path.join(hydra.utils.get_original_cwd(), cfg.eval_checkpoint)
    fout = fcheckpoint + '.pred.json'

    print('evaluating {}'.format(fcheckpoint))
    assert os.path.isfile(fcheckpoint)
    val = load_data(cfg.feval)
    Model = SupervisedModel.load_model_class(cfg.model)
    predictions = Model.run_inference(cfg, fcheckpoint=fcheckpoint, eval_dataset=val, test=False)

    print('wrote file to {}'.format(fout))
    with open(fout, 'wt') as f:
        json.dump(predictions, f, indent=2)


if __name__ == '__main__':
    main()
