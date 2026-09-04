"""EvalAI-compatible answer normalization and VQA soft accuracy.

The normalization rules follow the Apache-2.0 lmms-eval implementation, which
in turn copies the original Facebook MMF EvalAI answer processor. Keeping this
small dependency local makes every experiment reproducible offline.
"""

from __future__ import annotations

import re


class EvalAIAnswerProcessor:
    CONTRACTIONS = {
        "aint": "ain't", "arent": "aren't", "cant": "can't",
        "couldve": "could've", "couldnt": "couldn't",
        "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
        "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
        "hadnt": "hadn't", "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
        "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
        "hes": "he's", "howd": "how'd", "howll": "how'll",
        "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've",
        "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd",
        "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll",
        "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
        "mightnt've": "mightn't've", "mightn'tve": "mightn't've",
        "mightve": "might've", "mustnt": "mustn't", "mustve": "must've",
        "neednt": "needn't", "notve": "not've", "oclock": "o'clock",
        "oughtnt": "oughtn't", "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
        "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's",
        "shouldve": "should've", "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
        "somebodys": "somebody's", "someoned": "someone'd",
        "someoned've": "someone'd've", "someone'dve": "someone'd've",
        "someonell": "someone'll", "someones": "someone's",
        "somethingd": "something'd", "somethingd've": "something'd've",
        "something'dve": "something'd've", "somethingll": "something'll",
        "thats": "that's", "thered": "there'd", "thered've": "there'd've",
        "there'dve": "there'd've", "therere": "there're",
        "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
        "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're",
        "theyve": "they've", "twas": "'twas", "wasnt": "wasn't",
        "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've",
        "werent": "weren't", "whatll": "what'll", "whatre": "what're",
        "whats": "what's", "whatve": "what've", "whens": "when's",
        "whered": "where'd", "wheres": "where's", "whereve": "where've",
        "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've",
        "wholl": "who'll", "whos": "who's", "whove": "who've",
        "whyll": "why'll", "whyre": "why're", "whys": "why's",
        "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've",
        "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
        "yall'd've": "y'all'd've", "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've",
        "you'dve": "you'd've", "youll": "you'll", "youre": "you're",
        "youve": "you've",
    }
    NUMBER_MAP = {
        "none": "0", "zero": "0", "one": "1", "two": "2",
        "three": "3", "four": "4", "five": "5", "six": "6",
        "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    }
    ARTICLES = {"a", "an", "the"}
    PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
    COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
    PUNCTUATIONS = [
        ";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+",
        "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!",
    ]

    def __call__(self, item: str) -> str:
        item = item.lower().replace(",", "").replace("?", "").replace("'s", " 's")
        item = item.replace("\n", " ").replace("\t", " ").strip()
        out = item
        for punctuation in self.PUNCTUATIONS:
            if (
                punctuation + " " in item
                or " " + punctuation in item
                or self.COMMA_STRIP.search(item) is not None
            ):
                out = out.replace(punctuation, "")
            else:
                out = out.replace(punctuation, " ")
        out = self.PERIOD_STRIP.sub("", out)
        words = []
        for word in out.lower().split():
            word = self.NUMBER_MAP.get(word, word)
            if word not in self.ARTICLES:
                words.append(self.CONTRACTIONS.get(word, word))
        return " ".join(words)


def vqa_soft_accuracy(prediction: str, answers: list[str]) -> tuple[float, str]:
    """Return official leave-one-out VQA accuracy and normalized prediction."""

    processor = EvalAIAnswerProcessor()
    predicted = processor(prediction)
    ground_truth = [processor(answer) for answer in answers]
    if not ground_truth:
        return 0.0, predicted
    per_annotator = []
    for index in range(len(ground_truth)):
        matches = sum(
            answer == predicted
            for other_index, answer in enumerate(ground_truth)
            if other_index != index
        )
        per_annotator.append(min(1.0, matches / 3.0))
    return sum(per_annotator) / len(per_annotator), predicted
