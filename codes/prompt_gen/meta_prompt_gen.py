"""
Generate the meta-prompt which will be fed to ChatGPT and humans to generate the real text prompt which will be fed to
text-2-image models.
The generated meta-prompt will be based on the desired skill.
"""
import sys
import numpy as np
import random

sys.path.append('../eval_metrics/detection')
from scenarios_obj_map import ScenariosObjsMapping


class MetaPromptGen:
    def __init__(self, ann_path, label_space_path):
        self.possible_scenarios = None
        self.levels = ["easy", "medium", "hard"]
        self.styles = [["animation", "real", "sketch"],
                       ["sunny", "cloudy", "rainy"],
                       ["colored", "black and white"],
                       ["morning", "night"]]
        self.scenarios_objs_map = ScenariosObjsMapping(ann_path=ann_path, label_space_path=label_space_path)
        self.data_type = "unidet"  # ["coco", "unidet"]

        # Read Places365:
        places365 = open("places365.txt", "r")
        places365 = places365.readlines()
        self.places365 = [place.split('/')[2].split(" ")[0] for place in places365]

        # Read COCO:
        coco_classes = open("coco.txt", "r")
        coco_classes = coco_classes.readlines()
        self.coco_classes = [obj.strip('\n') for obj in coco_classes]

    def _select_rand_objs(self, num):
        objs = []
        for i in range(num):
            if self.data_type:
                obj = self._select_rand_obj_coco()
            else:
                obj = self._select_rand_obj()

            # check that there is no repeated objects:
            if objs:
                while obj in objs:
                    if self.data_type:
                        obj = self._select_rand_obj_coco()
                    else:
                        obj = self._select_rand_obj()

            objs.append(obj)
        return objs

    def _select_rand_obj(self):
        selected_scenario_objs = self.scenarios_objs_map.unidet_scenario_obj_map[random.choice(self.possible_scenarios)]
        obj = random.choice(self.scenarios_objs_map.unidet_categories_names[random.choice(selected_scenario_objs)])
        # check if it is valid object or not:
        if "--" in obj:
            obj = self._select_rand_obj()

        return obj

    def _select_rand_obj_coco(self):
        return random.choice(self.coco_classes)

    def _select_rand_style(self, num):
        sampled_styles = np.random.choice(a=self.styles, size=num, replace=False)
        sampled_styles = [random.choice(style) for style in sampled_styles]
        # handle the conflict between sunny and night:
        if (num > 1) and ("night" in sampled_styles) and ("sunny" in sampled_styles):
            sampled_styles = list(map(lambda x: x.replace('night', 'morning'), sampled_styles))

        return sampled_styles

    def select_rand_place(self):
        place = random.choice(self.places365).replace("_", " ")
        return place

    def _counting_gen(self):
        style = "real"
        self.data_type = "coco"  # ["coco", "unidet"]
        self.possible_scenarios = ["animals", "transportation", "home devices", "cleaning tools", "maintenance tools",
                                   "cooking tools", "furniture", "wild life", "toys", "musical instruments", "clothes",
                                   "military", "signs", "autonomous driving", "movie", "fashion", "teaching", "sports"]
        n1, n2 = 0, 0
        obj1, obj2 = None, None
        if self.level == "easy":
            n1 = np.random.random_integers(low=1, high=2)  # [1, 2]
            obj1 = self._select_rand_objs(num=1)[0]
            counting_template = "Describe a {style} scene about {N1} {obj1}.".format(style=style, N1=n1, obj1=obj1)
            vanilla_template = "{N1} {obj1}".format(N1=n1, obj1=obj1)
        elif self.level == "medium":
            n1 = np.random.random_integers(low=2, high=3)  # [2, 3]
            n2 = np.random.random_integers(low=2, high=3)  # [2, 3]
            objs = self._select_rand_objs(num=2)
            obj1, obj2 = objs
            counting_template = "Describe a {style} scene about {N1} {obj1} and {N2} {obj2}.".format(style=style,
                                                                                                     N1=n1, obj1=obj1,
                                                                                                     N2=n2, obj2=obj2)
            vanilla_template = "{N1} {obj1} and {N2} {obj2}".format(N1=n1, obj1=obj1, N2=n2, obj2=obj2)
        elif self.level == "hard":
            n1 = np.random.random_integers(low=4, high=5)  # [3, 4]
            n2 = np.random.random_integers(low=4, high=5)  # [3, 4]
            objs = self._select_rand_objs(num=2)
            obj1, obj2 = objs
            counting_template = "Describe a {style} scene about {N1} {obj1} and {N2} {obj2}.".format(style=style,
                                                                                                     N1=n1, obj1=obj1,
                                                                                                     N2=n2, obj2=obj2)
            vanilla_template = "{N1} {obj1} and {N2} {obj2}".format(N1=n1, obj1=obj1, N2=n2, obj2=obj2)
        else:
            raise Exception("Sorry, the selected level is not implemented, the only implemented options are ",
                            self.levels)
        return {"meta_prompt": "In one sentence, " + counting_template, "vanilla_prompt": vanilla_template,
                "n1": n1, "obj1": obj1, "n2": n2, "obj2": obj2}

    def _fidelity_gen(self):
        self.possible_scenarios = []
        obj1, obj2, obj3, style1, style2, style3 = None, None, None, None, None, None
        for k, v in self.scenarios_objs_map.unidet_scenario_obj_map.items():
            if v:
                self.possible_scenarios.append(k)
        if self.level == "easy":
            n1 = 1
            obj1 = self._select_rand_objs(num=1)[0]
            style = self._select_rand_style(num=1)[0]
            template = "Describe a {style} scene about {obj1}.".format(style=style, obj1=obj1)
        elif self.level == "medium":
            n1 = np.random.random_integers(low=1, high=2)  # [1, 2]
            n2 = np.random.random_integers(low=1, high=2)  # [1, 2]
            objs = self._select_rand_objs(num=2)
            obj1, obj2 = objs
            style1, style2 = self._select_rand_style(num=2)
            template = "Describe a {style1} {style2} scene about {obj1} and {obj2}.".format(style1=style1,
                                                                                            style2=style2,
                                                                                            obj1=obj1,
                                                                                            obj2=obj2)
        elif self.level == "hard":
            n1 = np.random.random_integers(low=1, high=2)  # [1, 2]
            n2 = np.random.random_integers(low=1, high=2)  # [1, 2]
            n3 = np.random.random_integers(low=1, high=2)  # [1, 2]
            style1, style2, style3 = self._select_rand_style(num=3)
            objs = self._select_rand_objs(num=3)
            obj1, obj2, obj3 = objs
            template = "Describe a {style1} {style2} {style3} scene about {obj1}, {obj2} and {obj3}.".format(
                style1=style1, style2=style2, style3=style3, obj1=obj1, obj2=obj2, obj3=obj3)
        else:
            raise Exception("Sorry, the selected level is not implemented, the only implemented options are ",
                            self.levels)
        return {"meta_prompt": "In one sentence, " + template, "obj1": obj1, "obj2": obj2, "obj3": obj3,
                "style1": style1, "style2": style2, "style3": style3}

    def _writing_gen(self):
        self.possible_scenarios = []
        obj1, obj2, n1 = None, None, 0
        for k, v in self.scenarios_objs_map.unidet_scenario_obj_map.items():
            if v:
                self.possible_scenarios.append(k)

        if self.level == "easy":
            objs = self._select_rand_objs(num=1)
            n1 = np.random.random_integers(low=1, high=3)  # [1, 3]
            template = "{N1} words about {obj1}, the {N1} words should be between double quotes.".format(
                obj1=objs[0], N1=n1)
        elif self.level == "medium":
            objs = self._select_rand_objs(num=2)
            n1 = np.random.random_integers(low=4, high=6)  # [4, 6]
            template = "{N1} words about {obj1} and {obj2}, the {N1} words should be between double quotes.".format(
                obj1=objs[0], obj2=objs[1], N1=n1)
        elif self.level == "hard":
            objs = self._select_rand_objs(num=2)
            n1 = np.random.random_integers(low=6, high=8)  # [6, 8]
            template = "{N1} words about {obj1} and {obj2}, the {N1} words should be between double quotes.".format(
                obj1=objs[0], obj2=objs[1], N1=n1)
        else:
            raise Exception("Sorry, the selected level is not implemented, the only implemented options are ",
                            self.levels)
        return {"meta_prompt": "one sentence of " + template, "n1": n1, "obj1": obj1, "obj2": obj2}

    def gen_meta_prompts(self, level_id, skill):
        self.level = self.levels[level_id]
        if skill == "fidelity":
            gen_meta_prompt = self._fidelity_gen()
        elif skill == "counting":
            gen_meta_prompt = self._counting_gen()
        elif skill == "writing":
            gen_meta_prompt = self._writing_gen()
        else:
            raise Exception("Sorry, the selected skill is not implemented")

        return gen_meta_prompt


if __name__ == "__main__":
    meta_prompt_gen = MetaPromptGen(ann_path="../../data/metrics/det/lvis_v1/lvis_v1_train.json",
                                    label_space_path="../eval_metrics/detection/UniDet-master/datasets/label_spaces/learned_mAP+M.json",
                                    )
    counting_template = meta_prompt_gen.gen_meta_prompts(level_id=0, skill="counting")
    print("counting_template: ", counting_template)
    print("Done !")
