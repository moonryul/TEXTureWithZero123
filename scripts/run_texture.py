import pyrallis
import sys
import os

# print(os.getcwd())
sys.path.append("./src/zero123/zero123")
sys.path.append("./src/zero123/ControlNet")

from src.configs.train_config import TrainConfig
from src.training.trainer import TEXTure

#MJ: when @pyrallis.wrap() is executed, it results in the execution of pyrallis.wrap(), which returns the wrapper_outer function. 
@pyrallis.wrap() #MJ:  It is designed to be used as a decorator on another function, main. @wrap(config_path="path/to/config.yaml")
def main(cfg: TrainConfig): #MJ: When wrapper_outer(main) is executed, it returns wrapper_inner.
# Inside wrapper_inner, the configuration cfg is obtained through the parse() function.
# The parse() function likely reads the configuration from a specified file path or uses default settings.
# The obtained cfg is then passed as the first argument to the original main function along with any other arguments and keyword arguments received by wrapper_inner.
    trainer = TEXTure(cfg)
    if cfg.log.eval_only:
        trainer.full_eval()
    else:
        trainer.paint()

#J: The wrapper_outer function is then used as the decorator. When another function, such as main(), is decorated with @pyrallis.wrap(), 
# it's actually equivalent to main = pyrallis.wrap()(main), 
# which means the main function is passed as an argument to the wrapper_outer function returned by pyrallis.wrap().
if __name__ == '__main__':
    main()
