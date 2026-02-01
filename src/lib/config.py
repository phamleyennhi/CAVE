# Copyright (c) Facebook, Inc. and its affiliates.
import collections
import logging
import os
from ast import literal_eval
import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='CAVE Configuration')
    parser.add_argument('--config', type=str, default='config/defaults.yaml')
    parser.add_argument('--cls_idx', type=int)
    parser.add_argument('--quantile', type=float)
    parser.add_argument('--normalize_relevance', type=int)
    parser.add_argument('--aggregate_by', type=str)
    parser.add_argument('--percentile', type=int)
    parser.add_argument('--merging_scheme', type=str)
    parser.add_argument('--linkage', type=str)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dataset', type=str, default="p3d_imagenet")
    parser.add_argument("--num_concepts", type=int, default=10)
    parser.add_argument("--model_architecture", type=str, default="resnetext")
    parser.add_argument("--occlusion", type=str, default="")
    parser.add_argument("--train", type=int, default=1)
    parser.add_argument("--seed_idx", type=int, default=None, help="Index of random seed (0-9) for k-means clustering")
    parser.add_argument("--concept_space", type=str)
    parser.add_argument("--shape", type=str)
    parser.add_argument("--concept_method", type=str, default=None)
    parser.add_argument("--target_img", type=str, default=None)
    parser.add_argument("--use_padding", type=int, default=0)
    return parser.parse_args()

def get_pkg_root():
    root = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(root))
    return root


def get_project_root():
    root = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(root, "../.."))
    return root


def get_abs_path(path):
    if not os.path.isabs(path):
        path = os.path.join(get_project_root(), path)
    return path


class ConfigNode(collections.OrderedDict):
    IMMUTABLE = "__is_frozen"

    def __init__(self, init_dict={}):
        self.__dict__[ConfigNode.IMMUTABLE] = False
        super().__init__(init_dict)

        for key in self:
            if isinstance(self[key], collections.abc.Mapping):
                self[key] = ConfigNode(self[key])
            elif isinstance(self[key], list):
                for idx, item in enumerate(self[key]):
                    if isinstance(item, collections.abc.Mapping):
                        self[key][idx] = ConfigNode(item)

    def freeze(self):
        for field in self.keys():
            if isinstance(self[field], collections.abc.Mapping):
                self[field].freeze()
            elif isinstance(self[field], list):
                for item in self[field]:
                    if isinstance(item, collections.abc.Mapping):
                        item.freeze()

        self.__dict__[ConfigNode.IMMUTABLE] = True

    def defrost(self):
        for field in self.keys():
            if isinstance(self[field], collections.abc.Mapping):
                self[field].defrost()
            elif isinstance(self[field], list):
                for item in self[field]:
                    if isinstance(item, collections.abc.Mapping):
                        item.defrost()

        self.__dict__[ConfigNode.IMMUTABLE] = False

    def __getattr__(self, key):
        if key not in self:
            raise AttributeError(key)

        return self[key]

    def __setattr__(self, key, value):
        if self.__dict__[ConfigNode.IMMUTABLE] is True:
            raise AttributeError("ConfigNode has been frozen and can't be updated")

        self[key] = value

    def _indent(self, st, num_spaces):
        st = st.split("\n")
        first = st.pop(0)
        st = [(num_spaces * " ") + line for line in st]
        st = [first] + st
        st = "\n".join(st)
        return st

    def __str__(self):
        strs = []

        if isinstance(self, collections.abc.Mapping):
            for key, value in sorted(self.items()):
                seperator = "\n" if isinstance(value, ConfigNode) else " "
                if isinstance(value, list):
                    attr_str = ["{}:".format(key)]
                    for item in value:
                        item_str = self._indent(str(item), 2)
                        attr_str.append("- {}".format(item_str))
                    attr_str = "\n".join(attr_str)
                else:
                    attr_str = "{}:{}{}".format(str(key), seperator, str(value))
                    attr_str = self._indent(attr_str, 2)
                strs.append(attr_str)
        return "\n".join(strs)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super().__repr__())

    def asdict(self):
        d = {}
        if isinstance(self, collections.abc.Mapping):
            for key, value in sorted(self.items()):
                if isinstance(value, ConfigNode):
                    d[key] = value.asdict()
                else:
                    d[key] = value
        return d


class Configuration:
    def __init__(self, config_yaml_file, default_config=None):
        self.config_path = config_yaml_file
        self.default_config = default_config
        self.config = {}

        base_config = {}
        if self.default_config is not None:
            base_config = self.load_yaml(self.default_config)

        user_config = self.load_yaml(self.config_path)

        self._base_config = base_config
        self._user_config = user_config

        self.config = self.nested_dict_update(base_config, user_config)

    def update_config(self, key, value):
        self.config[key] = value

    def get_config(self):
        return self.config

    def load_yaml(self, file):
        with open(file) as stream:
            mapping = yaml.safe_load(stream)

            if mapping is None:
                mapping = {}

            includes = mapping.get("includes", [])

            if isinstance(includes, str):
                includes = [includes]
            if not isinstance(includes, list):
                raise AttributeError(
                    "Includes must be a list or a str, {} provided".format(type(includes))
                )
            include_mapping = {}

            for include in includes:
                include = nemo.utils.get_abs_path(include)
                current_include_mapping = self.load_yaml(include)
                include_mapping = self.nested_dict_update(
                    include_mapping, current_include_mapping
                )

            mapping.pop("includes", None)

            mapping = self.nested_dict_update(include_mapping, mapping)

            return mapping

    def nested_dict_update(self, dictionary, update):
        """Updates a dictionary with other dictionary recursively.

        Parameters
        ----------
        dictionary : dict
            Dictionary to be updated.
        update : dict
            Dictionary which has to be added to original one.

        Returns
        -------
        dict
            Updated dictionary.
        """
        if dictionary is None:
            dictionary = {}

        for k, v in update.items():
            if isinstance(v, collections.abc.Mapping):
                dictionary[k] = self.nested_dict_update(dictionary.get(k, {}), v)
            else:
                dictionary[k] = self._decode_value(v)
        return dictionary

    def freeze(self):
        self.config = ConfigNode(self.config)
        self.config.freeze()
        return self

    def _merge_from_list(self, opts):
        if opts is None:
            opts = []

        assert len(opts) % 2 == 0, "Number of opts should be multiple of 2"

        for opt, value in zip(opts[0::2], opts[1::2]):
            splits = opt.split(".")
            current = self.config
            for idx, field in enumerate(splits):
                if field not in current:
                    raise AttributeError(
                        "While updating configuration"
                        " option {} is missing from"
                        " configuration at field {}".format(opt, field)
                    )
                if not isinstance(current[field], collections.abc.Mapping):
                    if idx == len(splits) - 1:
                        if nemo.utils.is_main_process():
                            logging.info(
                                "Overriding option {} to {}".format(opt, value)
                            )

                        current[field] = self._decode_value(value)
                    else:
                        raise AttributeError(
                            "While updating configuration",
                            "option {} is not present "
                            "after field {}".format(opt, field),
                        )
                else:
                    current = current[field]

    def override_with_cmd_opts(self, opts):
        self._merge_from_list(opts)

    def _decode_value(self, value):
        # https://github.com/rbgirshick/yacs/blob/master/yacs/config.py#L400
        if not isinstance(value, str):
            return value

        if value == "None":
            return None

        if value.endswith(".yaml"):
            return self.load_yaml(nemo.utils.get_abs_path(value))

        try:
            value = literal_eval(value)
        except ValueError:
            pass
        except SyntaxError:
            pass
        return value


def load_config(args, load_default_config=True, override=None, log_info=True):
    config_path = get_abs_path(args.config)
    assert os.path.isfile(
        config_path
    ), f"configuration file does not exist: {config_path}"

    if load_default_config:
        config = Configuration(
            config_path,
            default_config=os.path.join(get_project_root(), "config/defaults.yaml"),
        )
    else:
        config = Configuration(config_path)
    if override is not None:
        config.override_with_cmd_opts(override)
    config = config.freeze().get_config()

    if log_info:
        print_str = (
            "\n" + "=" * 16 + "\nConfigurations\n" + "=" * 16 + "\n" + str(config)
        )
        logging.info(print_str)

    config.defrost()
    config.args = args

    config.freeze()

    return config
