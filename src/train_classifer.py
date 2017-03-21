import sys
sys.path.append('./src/common')
sys.path.append('./experiment_settings')
from classification_settings import get_args
from trainer_utils import EasyTrainer


if __name__ == '__main__':
    print("-------traing")
    args = get_args('train')
    settings_type = 'classification_settings'
    e_trainer = EasyTrainer(args, settings_type)
    e_trainer.run_trainer()
