from hypothesis.strategies import composite, sampled_from, tuples, lists

__all__ = ['gender_cats', 'browser_cats', 'from_cats', 'cats', 'sentence',
        'corpus_strategy', 'sentence_strategy']

gender_cats = ['male', 'female']
browser_cats = ['Safari', 'IE', 'Firefox']
from_cats = ["Europe", "US", "Japan"]

@composite
def cats(draw, max_size=1):
    gender = sampled_from(gender_cats)
    browser = sampled_from(browser_cats)
    from_ = sampled_from(from_cats)
    sample = tuples(gender, browser, from_)

    return draw(lists(sample, max_size=max_size, min_size=1))


sentence = 'it was the best of times it was the blurst of times'
def corpus_strategy():
    return lists(min_size=2, 
            elements=sentence_strategy(),
            )

def sentence_strategy():
    return lists(elements=sampled_from(sentence.split()), min_size=4).map(lambda x: " ".join(x))
