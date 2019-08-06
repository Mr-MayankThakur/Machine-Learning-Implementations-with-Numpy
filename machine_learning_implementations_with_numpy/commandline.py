import click

import machine_learning_implementations_with_numpy.serve as serve_module
import machine_learning_implementations_with_numpy.train as train_module
import machine_learning_implementations_with_numpy.prepare as prepare_module


@click.group()
def cli():
    pass


@cli.command(help='Prepares the data for training')
def prepare():
    prepare_module.init()
    prepare_module.run()


@cli.command(help='Trains the model')
def train():
    train_module.init()
    train_module.run()


@cli.command(help='Serves the trained model')
def serve():
    serve_module.init()
    serve_module.run()
