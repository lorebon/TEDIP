"""Choose the rule budget by mean training-fold validation accuracy."""

from cli import main


if __name__ == "__main__":
    main(validate=True)
