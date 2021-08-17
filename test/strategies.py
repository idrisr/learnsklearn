from hypothesis.strategies import composite, sampled_from, tuples, lists

__all__ = ['gender_cats', 'browser_cats', 'from_cats', 'cats']

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
