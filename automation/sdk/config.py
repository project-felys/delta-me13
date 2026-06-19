from typing import Callable

from automation.api.sentence import Sentence


def sentence_global_config(
    auto_format: Callable[[Sentence], str] | None,
    token_counter: Callable[[str], int] | None,
    match_sub: Callable[[str], str] | None,
):
    if auto_format is not None:
        Sentence.set_auto_format_func(auto_format)
    else:
        Sentence.set_auto_format_func(lambda x: x.text)

    if token_counter is not None:
        Sentence.set_token_counter_func(token_counter)
    else:
        Sentence.set_token_counter_func(lambda x: len(x))

    if match_sub is not None:
        Sentence.set_match_sub_func(match_sub)
    else:
        Sentence.set_match_sub_func(lambda x: x)
