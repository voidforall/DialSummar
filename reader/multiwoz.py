import json
import logging
from typing import List, Dict

import numpy as np
import re
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, MultiLabelField, ListField, ArrayField, MetadataField, NamespaceSwappingField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)

class MultiwozReader(DatasetReader):
    """
    Reads a json file containing the preprocessed data from MultiWOZ2.0
    MultiWOZ: "MultiWOZ - A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling"
    (arXiv: https://arxiv.org/abs/1810.00278)

    Expected format of an Instance in .json file:
    id: {
        "usr": ["sent1, sent2..."],
        "sys": ["sent1, sent2..."],
        "target": ["sent1", sent2..."],
        "domains": [domain1, domain2...], ( >= 1 domain
        ([restaurant, hotel, attraction, taxi, train, hospital, police])
    }
    The other two fields "values" and "values_gt" is related to the delexicalized slot values
    in source text and target text.

    The output of "read" is a list of Instances with the following fields:
    user: ListField[TextField]
    sys: ListField[TextField]
    target: TextField (=None for evaluation & No ListField for decoding token by token)
    domains: MultiLabelField
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, Tokenizer] = None,
                 target_namespace: str = "target_tokens", # somewhat necessary to use two namespace in Vocab
                 lazy: bool = False
                 ) -> None:
        super().__init__(lazy)
        # JustSpacesWordSplitter can maintain [value_slot] in dataset and our data has been tokenized
        self._target_namespace = target_namespace # need to separate namespaces of source & target
        self._tokenizer = tokenizer or WordTokenizer(word_splitter=JustSpacesWordSplitter())
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexer = {
            "tokens": SingleIdTokenIndexer(namespace=self._target_namespace)
        }

    # _tokens_to_ids: setdefault just functions as allocating ids of tokens sequentially
    # also, the id is not equal with the index in the vocabulary
    @staticmethod
    def _tokens_to_ids(tokens: List[Token]) -> List[int]:
        ids: Dict[str, int] = {}
        out: List[int] = []
        for token in tokens:
            out.append(ids.setdefault(token.text.lower(), len(ids)))
        return out

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from file at %s", file_path)
            data_json = json.load(data_file)
            for k, v in data_json.items():
                user = v["usr"]
                system = v["sys"]
                target = v["target"]
                domains = v["domains"]
                values = v["values"]
                
                # match the `values` with token index (separately)
                usr_value_dict = {}
                sys_value_dict = {}
                usr_index = 1
                sys_index = 1
                for turn in range(len(user)):
                    regex = re.compile("\s+")
                    user_turn = regex.split(user[turn])
                    for token in user_turn:
                        if token == "":
                            continue
                        if token[0] == "[":
                            usr_value_dict[usr_index] = values[0]
                            del values[0]
                        usr_index += 1
                    
                    sys_turn = regex.split(system[turn])
                    for token in sys_turn:
                        if token == "":
                            continue
                        if token[0] == "[":
                            sys_value_dict[sys_index] = values[0]
                            del values[0]
                        sys_index += 1

                # use Iterator for laziness
                yield self.text_to_instance(user, system, domains, usr_value_dict, sys_value_dict, target)

    @overrides
    def text_to_instance(self, user: List[str], system: List[str],
                         domains: List[str],
                         usr_value_dict: Dict[int, str],
                         sys_value_dict: Dict[int, str],
                         # acts: List[List[str]],
                         target: List[str]=None) -> Instance:

        fields_dict: Dict[str, Field] = {}

        # Note that it's a non-hierarchical model,
        # so the user/system/target are all TextField
        user_string = " ".join(user)
        tokenized_user = self._tokenizer.tokenize(user_string)
        tokenized_user.insert(0, Token(START_SYMBOL))
        tokenized_user.append(Token(END_SYMBOL))
        user_field = TextField(tokenized_user, self._token_indexers)
        fields_dict["user_tokens"] = user_field

        sys_string = " ".join(system)
        tokenized_sys = self._tokenizer.tokenize(sys_string)
        tokenized_sys.insert(0, Token(START_SYMBOL))
        tokenized_sys.append(Token(END_SYMBOL))
        sys_field = TextField(tokenized_sys, self._token_indexers)
        fields_dict["sys_tokens"] = sys_field

        # For each token in the source sentence, we keep track of the matching token
        # in the target sentence (which will be the OOV symbol) if there is no match
        # p.s. separate matching of user and system. wait, do they need separate namespace?
        user_to_target_field =NamespaceSwappingField(tokenized_user, self._target_namespace)
        sys_to_target_field = NamespaceSwappingField(tokenized_sys, self._target_namespace)
        fields_dict["user_to_target"] = user_to_target_field
        fields_dict["sys_to_target"] = sys_to_target_field

        meta_fields = {"user_tokens": [x.text for x in tokenized_user],
                       "sys_tokens": [x.text for x in tokenized_sys],
                       "user_values_dict": usr_value_dict,
                       "sys_values_dict": sys_value_dict}
        
        # add: generate the mask of "delex" slots
        usr_mask = np.zeros(len(user_field))
        for k in usr_value_dict.keys():
            usr_mask[k] = 1
        fields_dict["user_value_mask"] = ArrayField(usr_mask)
        sys_mask = np.zeros(len(sys_field))
        for k in sys_value_dict.keys():
            sys_mask[k] = 1
        fields_dict["sys_value_mask"] = ArrayField(sys_mask)
        
        if target is not None:
            target_string = " ".join(target)
            tokenized_target = self._tokenizer.tokenize(target_string)
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, self._target_token_indexer)

            fields_dict["target_tokens"] = target_field
            meta_fields["target_tokens"] = [y.text for y in tokenized_target]

            user_token_ids = self._tokens_to_ids(tokenized_user)
            sys_token_ids = self._tokens_to_ids(tokenized_sys)
            target_token_ids = self._tokens_to_ids(tokenized_target)
            fields_dict["user_token_ids"] = ArrayField(np.array(user_token_ids))
            fields_dict["sys_token_ids"] = ArrayField(np.array(sys_token_ids))
            fields_dict["target_token_ids"] = ArrayField(np.array(target_token_ids))

        else:
            user_token_ids = self._tokens_to_ids(tokenized_user)
            sys_token_ids = self._tokens_to_ids(tokenized_sys)
            fields_dict["user_token_ids"] = ArrayField(np.array(user_token_ids))
            fields_dict["sys_token_ids"] = ArrayField(np.array(sys_token_ids))

        domain_field = MultiLabelField(domains, label_namespace="domain_labels")
        fields_dict["domain_labels"] = domain_field

        fields_dict["metadata"] = MetadataField(meta_fields)

        return Instance(fields_dict)