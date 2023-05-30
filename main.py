from options import Options
from trainer import TrainManager
from inference import InferenceManager

if __name__ == '__main__':
    print('Executing main')

    OPTIONS = Options()
    OPTIONS = OPTIONS.parse()

    if OPTIONS.mode == 'train':
        print('In training mode!')
        TRAINER = TrainManager(OPTIONS)
        TRAINER.train()

    elif OPTIONS.mode == 'inference':
        print('In inference mode!')
        TESTER = InferenceManager(OPTIONS)
        TESTER.run_inference()

    else:
        raise NotImplementedError
