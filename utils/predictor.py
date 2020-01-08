from overrides import overrides
import re

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

class MultiwozPredictor(Predictor):
    """
    Predictor wrapper which is applied on evaluation of testing set.
    """
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        return output_dict

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        user = json_dict["usr"]
        system = json_dict["sys"]
        domains = json_dict["domains"]
        values = json_dict["values"]
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
        
        return self._dataset_reader.text_to_instance(user, system, domains, usr_value_dict, sys_value_dict)