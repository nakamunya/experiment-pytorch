import argparse
import torch
import yaml

from torch.optim import SGD
import torch.nn.functional as F

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.metrics import Accuracy, Loss

from tqdm import tqdm

from mssd.models.sample_net import SampleNet
from mssd.data import get_data_loaders


def score_function(engine):
    val_loss = engine.state.metrics['nll']
    return -val_loss


def train(cfg, model, train_loader, val_loader, optimizer, device):
    trainer = create_supervised_trainer(model, optimizer, F.nll_loss, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': Accuracy(),
                                                    'nll': Loss(F.nll_loss)},
                                            device=device)

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0)
    )

    @trainer.on(Events.ITERATION_COMPLETED(every=cfg["log"]["interval"]))
    def log_training_loss(engine):
        pbar.desc = desc.format(engine.state.output)
        pbar.update(cfg["log"]["interval"])

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        tqdm.write(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll)
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        tqdm.write(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll))

        pbar.n = pbar.last_print_n = 0

    # # Checkpoint setting
    # ./checkpoints/sample_mymodel_{step_number}
    handler = ModelCheckpoint(dirname=cfg["checkpoint"]["path"], filename_prefix=cfg["checkpoint"]["prefix"], n_saved=3, create_dir=True, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=2), handler, {'mymodel': model})

    # # Early stopping
    handler = EarlyStopping(patience=5, score_function=score_function, trainer=trainer)
    # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset)
    evaluator.add_event_handler(Events.COMPLETED, handler)

    trainer.run(train_loader, max_epochs=cfg["training"]["epoch"])
    pbar.close()


def main(cfg):
    train_loader, val_loader = get_data_loaders(cfg["training"]["batch_size"],
                                                cfg["validate"]["batch_size"])
    model = SampleNet()
    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'

    optimizer = SGD(model.parameters(),
                    lr=cfg["training"]["lr"],
                    momentum=cfg["training"]["momentum"])

    train(cfg, model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, device=device)

    # # Save model
    torch.save(model.state_dict(), './checkpoints/final_weights.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/debug.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    main(cfg)
