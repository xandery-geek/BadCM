from transformers import BertConfig, BertTokenizer, AutoModelForMaskedLM
from badcm.modules.lazy_loader import LazyLoader


hub = LazyLoader("tensorflow_hub", globals(), "tensorflow_hub")

class TextualGenertor(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg

        self.mlm_model, self.tokenizer = self.load_model()
        self.sentence_encoder = hub.load('https://tfhub.dev/google/universal-sentence-encoder/3')


    def load_data(split):
        # load text

        # load critical words

        pass

    def load_model(self):
        mlm_path = self.cfg['mlm_path']

        config_atk = BertConfig.from_pretrained(mlm_path)
        mlm_model = AutoModelForMaskedLM.from_pretrained(mlm_path, config=config_atk)
        tokenizer = BertTokenizer.from_pretrained(mlm_path, do_lower_case=True)

        return mlm_model, tokenizer

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