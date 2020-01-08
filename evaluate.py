import numpy as np
import json
import re

valid_list = ["taxi_arriveBy", "taxi_leaveAt", "taxi_destination", "taxi_departure",
             "hotel_name", "hotel_type", "hotel_area", "hotel_pricerange",
             "hotel_stars", "hotel_people", "hotel_day", "hotel_stay",
             "attraction_type", "attraction_name", "attraction_area",
             "train_leaveAt", "train_arriveBy", "train_departure", "train_destination", "train_day", "train_people",
             "restaurant_name", "restaurant_food", "restaurant_pricerange", "restaurant_area",
             "restaurant_people", "restaurant_day", "restaurant_time"]

# evaluate slot-value recall based on the reqt_dict.json and prediction.txt
# note: we may skip some slots which are less important
# regular expression to be matched needs to be "least matching", such as "friday", "6 people"...
reqt_dict_path = "./data/reqt_dict.json"
reqt_dict = json.load(open(reqt_dict_path))
test_path = "./data/test_delex.json"
test_json = json.load(open(test_path))
test_keys = [k for k in test_json.keys()]

def evaluate(prediction_path):
    predictions = open(prediction_path, "r+")
    prediction_list = []
    for line in predictions:
        prediction_list.append(line)
    
    final_score = 0.0
    
    for i in range(100):
        test_key = test_keys[i]
        reqt = reqt_dict[test_key]
        prediction_text = prediction_list[i] # just a string
        
        domain_num = 0
        recall_score = 0.0
        for k, v in reqt.items():
            # focus on this domain
            if v:
                domain_num += 1
                count_match = 0
                count_all = 0
                for domain_type, type_dict in v.items():
                    if domain_type == "reqt":
                        continue
                    if type(type_dict) != dict:
                        continue
                    for slot, value in type_dict.items():
                        search_slot = k + "_" + slot
                        if search_slot in valid_list:
                            if re.findall(value, prediction_text):
                                count_match += 1
                                count_all += 1
                            else:
                                count_all += 1
                if count_all != 0:
                    domain_recall = count_match / count_all
                    recall_score += domain_recall
        recall_score = recall_score / domain_num
        # print(recall_score)
        final_score += recall_score
    print("The recall score of entities: ", final_score / 100)