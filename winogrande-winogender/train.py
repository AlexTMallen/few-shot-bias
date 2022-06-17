#!/usr/bin/env python
import hydra
import logging
import ujson as json
from wrangl.learn import SupervisedModel


logger = logging.getLogger(__name__)


def load_data(fname):
    return [json.loads(x) for x in open(fname)]


@hydra.main(config_path='conf', config_name='default')
def main(cfg):
    train = load_data(cfg.ftrain)
    val = load_data(cfg.feval)
    Model = SupervisedModel.load_model_class(cfg.model)
    Model.run_train_test(cfg, train, val)


if __name__ == '__main__':
    main()
