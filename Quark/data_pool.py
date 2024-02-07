from typing import List
from copy import deepcopy


class DataPool:
    def __init__(self, tree_tokens, n_extra_tokens, only_top=False):
        self.tree_tokens = tree_tokens
        self.n_extra_tokens = n_extra_tokens

        self.cat_tokens = None
        self.prompt_pool, self.response_pool, self.score_pool = [], [], []

        self.only_top = only_top

        self.best_cat = self.tree_tokens[0]

    def add(self, prompts: List[str], responses: List[str], scores: List[float]):
        self.prompt_pool.extend(prompts)
        self.response_pool.extend(responses)
        self.score_pool.extend(scores)

        data = zip(self.prompt_pool, self.response_pool, self.score_pool)
        data = [x for x in data if x[-1] is not None]
        sorted_data = sorted(data, key=lambda x: x[-1], reverse=True)
        self.prompt_pool, self.response_pool, self.score_pool = [list(x) for x in list(zip(*sorted_data))]

        cat_pos = [[i] * (len(sorted_data) // self.n_extra_tokens) for i in range(self.n_extra_tokens)]
        cat_pos = [y for x in cat_pos for y in x]
        cat_pos = cat_pos + [self.n_extra_tokens - 1] * (len(sorted_data) - len(cat_pos))
        self.cat_tokens = [self.tree_tokens[i] for i in cat_pos]

        if self.only_top:
            assert len(self.prompt_pool) == len(self.response_pool), "sizes don't match"
            assert len(self.prompt_pool) == len(self.score_pool), "sizes don't match"
            assert len(self.prompt_pool) == len(self.cat_tokens), "sizes don't match"
            zipped_data = zip(self.prompt_pool, self.response_pool, self.score_pool, self.cat_tokens)

            #print(zipped_data)
            #print(self.cat_tokens)
            #print(self.best_cat)
            zipped_data = [x for x in zipped_data if x[-1] == self.best_cat]
            #print(zipped_data)
            self.prompt_pool, self.response_pool, self.score_pool, self.cat_tokens = [list(x) for x in list(zip(*zipped_data))]
            #for p,r,c in zip(self.prompt_pool, self.response_pool, self.cat_tokens):
            #    if c == 

    def get_data(self):
        return deepcopy(self.prompt_pool), deepcopy(self.response_pool), deepcopy(self.cat_tokens)

