import re
import urllib.request


def get_vocab():
    url = ("https://raw.githubusercontent.com/rasbt/"
          "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
          "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("Total number of characters: ", len(raw_text))
    print(raw_text[:99])

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print("Preprocessed items: ", len(preprocessed))
    print(preprocessed[:30])

    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])

    vocab_size = len(all_tokens)
    print("Vocabulary size: ", vocab_size)

    print("Vocabulary: ")
    vocab = {token:i for i, token in enumerate(all_tokens)}
    for i, item in enumerate(vocab.items()):
        print(item)
        if i >= 50:
            break
    for i, item in enumerate(list(vocab.items())[-5:]):
        print(item)
        
    return vocab


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)   
        return text

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        # change in v2
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # change in v2: replaces spaces before the specified puntuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

def main():
    print("Vocabulary:")
    vocab = get_vocab()

    print("Tokenizer V1:")
    tokenizer = SimpleTokenizerV1(vocab)
    text = """It's the last he painted, you know," 
       Mrs. Gisburn said with pardonable pride."""

    print("Encode:")
    ids = tokenizer.encode(text)
    print("IDs: ", ids)

    print("Decode:")
    texts = tokenizer.decode(ids)
    print("Texts: ", texts)

    print("Tokenizer V2:")
    tokenizer = SimpleTokenizerV2(vocab)
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))

    print("Encode:")
    ids = tokenizer.encode(text)
    print("IDs: ", ids)

    print("Decode:")
    texts = tokenizer.decode(ids)
    print("Texts: ", texts)

if __name__ == "__main__":
    main()
