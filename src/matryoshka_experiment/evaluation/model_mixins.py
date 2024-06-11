from sentence_transformers import SentenceTransformer


class PrependPromptMixin:
    def __init__(self, prompt: str, model: SentenceTransformer, debug: bool = False):
        self.prompt = prompt
        self.model = model
        self.debug = debug

    def encode(self, sentences, ** kwargs):
        if self.debug:
            print(f"Prepending prompt to the first example:\n{sentences[0]}", "\n\n\n", f"Examples to the model goes like this:\n{self.prompt}: {sentences[0]}")
        # TODO: lets define format of prompt
        return self.model.encode([f"{self.prompt}: {sentence}" for sentence in sentences], **kwargs)

