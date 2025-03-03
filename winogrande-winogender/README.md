# XOR classifier

This example trains a Winogrande model.

```bash
git clone https://github.com/AlexTMallen/few-shot-bias
cd winogrande-winogender
```

## Training
To train locally (settings in `conf/default.yaml`):

```bash
python train.py
```

To train using Slurm via the Slurm launcher:

```bash
python train.py --multirun hydra/launcher=slurm hydra.launcher.partition=<name of your partition>
```

## Git
To track code changes on a run-to-run basis:

```bash
python train.py git.enable=true
```

This will save Git diffs to the run output directory in `saves/<project name>/<run name>/{git.head.json, git.path.diff}`.


## WanDB
To log results onto Wandb (assuming your `wandb` user is your current `$USER`, if not, you can specify `wandb.entity=<your wandb username>`):

```bash
python train.py wandb.enable=true
```

[Here is an example](https://wandb.ai/vzhong/wrangl-examples-xor_clf) of the Wandb run for this job.

## S3
To log results into your custom S3 bucket, first create a JSON file (e.g. `mycredentials.json`) with your S3 credentials:

```json
{
  "WRANGL_S3_URL": "s3.myminio.com",
  "WRANGL_S3_BUCKET": "wrangl",
  "WRANGL_S3_KEY": "mykey",
  "WRANGL_S3_SECRET": "mysecret"
}
```

Then run

```bash
python train.py s3.enable=true s3.fcredentials=mycredentials.json
```

This will upload the experiment to your S3 bucket at `https://s3.myminio.com/wrangl/<project name>/<run name>/`.
This will contain the experiment configs, the best checkpoint, training logs, and Git patches.
Additionally, it will plot the metrics over time by default and save that to a PDF.
