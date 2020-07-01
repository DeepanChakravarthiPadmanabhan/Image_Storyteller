import nltk

def compute_BLEU_score(reference, hypothesis):
    BLEU_1 = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(1.0, 0.0, 0.0, 0.0))
    BLEU_2 = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0.5, 0.5, 0.0, 0.0))
    BLEU_3 = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0.33, 0.33, 0.33, 0.0))
    BLEU_4 = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0.25, 0.25, 0.25, 0.25))
    return BLEU_1, BLEU_2, BLEU_3, BLEU_4