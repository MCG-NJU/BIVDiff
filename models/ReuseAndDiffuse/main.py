import logging

import hydra
from hydra.utils import instantiate

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="example", version_base=None)
def main(config):
    model = instantiate(config.model)
    model.setup(stage="test")
    evaluator = instantiate(config.evaluator)
    evaluator(model)
    logger.info("finished.")


if __name__ == "__main__":
    main()
