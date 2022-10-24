

class TextualGenertor(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def load_data():
        # load text

        # load critical words

        pass

    def load_model():
        pass

    def get_substitues():
        pass

    def generate_poisoned_txt(split):
        pass


def run(cfg):
    module = TextualGenertor(cfg)

    module.generate_poisoned_txt('train')
    module.generate_poisoned_txt('test')


if __name__ == "__main__":
    cfg = {

    }
    
    run(cfg)